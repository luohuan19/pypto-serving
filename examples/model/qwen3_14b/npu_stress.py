# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Batched stress + profiling run for Qwen3-14B, mirroring the vLLM-Ascend
benchmark layout: a ~3.3k-token LONG_PROMPT replicated across a 16-way batch
with 128 output tokens, reported as prefill / decode throughput.

Example:
    ts --env SIMPLER_CHIP_TIMING=1 --env PTO2_RING_HEAP=4294967296 \
       --env PTO2_RING_TASK_WINDOW=131072 --env PTO2_RING_DEP_POOL=131072 \
       "python examples/model/qwen3_14b/npu_stress.py \
         --model-dir /data/linyifan/models/Qwen3-14B \
         --platform a2a3 --batch-size 16 --max-new-tokens 128 \
         --max-seq-len 4096 --device-id {} &> stress.log"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Cap CPU BLAS threads BEFORE torch/numpy import their BLAS backend. On big-core
# hosts (e.g. 320 cores) OpenBLAS oversubscribes past its precompiled thread limit
# and thrashes ("BLAS : Bad memory unallocation" flood) — hanging the large matmuls
# the vectorized CPU prefill issues. Only affects host-side BLAS (the NPU kernels
# are unaffected). Override via `--env OPENBLAS_NUM_THREADS=...` if needed.
_BLAS_THREADS = str(min(64, os.cpu_count() or 64))
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _BLAS_THREADS)


def _bootstrap_package_root() -> None:
    this_file = Path(__file__).resolve()
    for candidate in (this_file, *this_file.parents):
        if (candidate / "python" / "core").is_dir() and (candidate / "examples" / "model" / "qwen3_14b" / "runner").is_dir():
            repo_root = str(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return
    raise RuntimeError(f"Unable to locate the pypto-serving repo root from {this_file}")


_bootstrap_package_root()

# Reuse the profiling helpers from the single-prompt script.
from examples.model.qwen3_14b.npu_generate import (  # noqa: E402
    InstallProfiling,
    PrintTimingReport,
    _TimingCollector,
    _install_num_layers_override,
)

from python.core import GenerateConfig, LLMEngine, RuntimeConfig  # noqa: E402
from python.core.kv_cache import KvCacheManager  # noqa: E402
from python.profile import get_profiler, merge_profile, profile_span  # noqa: E402
from examples.model.qwen3_14b.runner.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor  # noqa: E402

_DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "long_prompt.txt"


def _enable_l3_scope_stats(out_dir: Path) -> None:
    """Turn on per-scope ring-fill (heap / task_window / dep_pool) capture on the
    L3 DistributedWorker path.

    Serving dispatches through ``pypto.runtime.distributed_runner._make_call_config``,
    which only overlays ring sizing onto the per-dispatch ``CallConfig`` and drops
    the DFX flags. We wrap it so every CallConfig it builds also carries
    ``enable_scope_stats`` + ``output_prefix``; the runtime then streams
    ``<out_dir>/scope_stats/scope_stats.jsonl`` (metadata line holds heap_max, so
    you can read peak heap_end vs the PTO2_RING_HEAP cap). Render with
    pypto-github/runtime/simpler_setup/tools/scope_stats_plot.py.

    Must run BEFORE the worker is constructed (i.e. before init_model), because the
    baseline CallConfig is built once at prepare() time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    from pypto.runtime import distributed_runner  # noqa: PLC0415

    orig = distributed_runner._make_call_config

    def patched(dc, run_config=None):  # type: ignore[no-untyped-def]
        call_config = orig(dc, run_config)
        call_config.enable_scope_stats = True
        call_config.output_prefix = str(out_dir)
        return call_config

    distributed_runner._make_call_config = patched
    print(f"[scope-stats] enabled; jsonl -> {out_dir}/scope_stats/scope_stats.jsonl", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batched Qwen3-14B stress test (vLLM-Ascend style report)."
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory (HF snapshot).")
    parser.add_argument(
        "-p",
        "--prompt-file",
        default=os.environ.get("PROMPT_FILE", str(_DEFAULT_PROMPT_FILE)),
        help="Text file holding the LONG_PROMPT. 优先级：-p/--prompt-file > 环境变量 "
             "PROMPT_FILE > 默认的同目录 long_prompt.txt（~3.3k tokens）。",
    )
    parser.add_argument("--model-id", default="qwen3-14b-local")
    parser.add_argument("--platform", default="a2a3", choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16, help="Concurrent requests (prompts) per batch.")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Runtime/kernel batch size. Must equal the compiled decode kernel BATCH (16).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of back-to-back batches to run (total requests = batch-size * num-batches).",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help="Split prefill into windows of this many tokens (bounds per-ring "
             "heap/task-window load). Default: prefill the whole prompt at once.",
    )
    parser.add_argument(
        "--prefill-on-cpu",
        action="store_true",
        help="Run prefill on host (torch reference) and push its KV into the device "
             "pool the decode kernel reads, instead of dispatching the NPU prefill "
             "kernel. Sidesteps the 40-layer prefill ring-arena OOM; decode stays on "
             "NPU. Much slower per prefill (minutes) — for offline / OOM-blocked runs.",
    )
    parser.add_argument(
        "--prefill-cache-dir",
        default=None,
        help="Cache CPU-prefill results (logical KV + first-token logits) under this "
             "dir and reuse them on a later run with the same prompt — skips the "
             "minutes-long host loop on a cache hit. Only effective with --prefill-on-cpu.",
    )
    parser.add_argument(
        "--dump-kv",
        default=None,
        metavar="PATH",
        help="Debug: after prefill, read each request's prompt KV back from the device "
             "pool and save logical K/V (+ token ids) to PATH (.pt). Run once with "
             "--prefill-on-cpu and once without, then compare to validate CPU-prefill KV.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--save-kernels-dir", default=None)
    parser.add_argument("--num-layers-override", type=int, default=None)
    parser.add_argument(
        "--scope-stats-dir",
        default=None,
        help="Enable per-scope ring-fill (heap/task_window/dep_pool) capture and "
             "write scope_stats/scope_stats.jsonl under this directory.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print phase / executor-API / per-kernel timing summary at the end.",
    )
    parser.add_argument(
        "--profile-verbose",
        action="store_true",
        help="Implies --profile. Also dump per-layer prefill and per-decode-step breakdowns.",
    )
    return parser


class _StepMeter:
    """Per-step host/device timing, mirroring the trace-based 关键指标 that
    vllm_stress.py derives from the Ascend Profiler kernel_details.csv.

    PyPTO already hands us the numbers vLLM had to dig out of a trace: the L3
    worker's ``timing.device_wall_us`` is the real on-device compute span of one
    dispatch, so we don't need to parse a profiler CSV. We capture, per decode
    step: the host wall around ``run_decode`` (= 步周期 / TPOT 真实单步) and the
    kernel's ``device_wall`` (= device 计算); ``host 间隙`` is their difference.
    """

    def __init__(self) -> None:
        self.decode_periods_ms: list[float] = []   # host wall around run_decode (TPOT)
        self.decode_device_ms: list[float] = []    # device_wall of the decode dispatch
        self.prefill_host_ms: list[float] = []      # host wall around run_prefill


def InstallStepMeter(engine: LLMEngine, model_id: str) -> _StepMeter:
    """Wrap executor dispatch + the runner's L3 dispatch to collect per-step
    host periods and device wall. Always-on (independent of --profile); composes
    on top of InstallProfiling when that is also installed. Run AFTER init_model.
    """
    meter = _StepMeter()
    executor = engine._executor  # type: ignore[attr-defined]
    compiled = executor._compiled[model_id]  # type: ignore[attr-defined]
    runner = executor._runners[model_id]  # type: ignore[attr-defined]

    decode_id = id(compiled.decode)
    prefill_id = id(compiled.prefill)
    orig_run_program = runner._run_distributed_program  # type: ignore[attr-defined]

    def timed_run_program(callable_spec, *args, **kwargs):
        timing = orig_run_program(callable_spec, *args, **kwargs)
        dev_us = getattr(timing, "device_wall_us", None)
        if dev_us is not None and id(callable_spec) == decode_id:
            meter.decode_device_ms.append(float(dev_us) / 1000.0)
        return timing

    runner._run_distributed_program = timed_run_program  # type: ignore[attr-defined]

    orig_prefill = executor.run_prefill
    orig_decode = executor.run_decode

    def timed_prefill(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_prefill(*args, **kwargs)
        finally:
            meter.prefill_host_ms.append((time.perf_counter() - t0) * 1000.0)

    def timed_decode(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_decode(*args, **kwargs)
        finally:
            meter.decode_periods_ms.append((time.perf_counter() - t0) * 1000.0)

    executor.run_prefill = timed_prefill
    executor.run_decode = timed_decode
    return meter


def _install_kv_dump(engine: LLMEngine, model_id: str, path: str) -> None:
    """Wrap run_prefill to dump the prompt KV (device read-back) once, right after
    the first prefill returns and before decode appends new-token KV. Run AFTER
    init_model; composes on top of InstallStepMeter's wrapper."""
    executor = engine._executor  # type: ignore[attr-defined]
    runner = executor._runners[model_id]  # type: ignore[attr-defined]
    orig_prefill = executor.run_prefill
    state = {"done": False}

    def dumping_prefill(model, batch, *args, **kwargs):
        result = orig_prefill(model, batch, *args, **kwargs)
        if not state["done"]:
            runner.dump_request_kv(model, batch, path)
            state["done"] = True
            print(f"[dump-kv] wrote prompt KV -> {path}", flush=True)
        return result

    executor.run_prefill = dumping_prefill


def _pct(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数，q ∈ [0, 1]，输入需已排序。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _print_report(
    *,
    batch_size: int,
    num_batches: int,
    prompt_tokens: int,
    output_tokens: int,
    per_request_latencies: list[float],
    total_elapsed: float,
    meter: _StepMeter,
) -> None:
    total_requests = batch_size * num_batches
    total_out = total_requests * output_tokens

    print("\n=== PyPTO 压力测试 + Profiling ===")
    print(f"  模型: qwen3-14b  批次: {batch_size}  总请求: {total_requests}")
    print(f"  输入: ~{prompt_tokens} tokens  输出: {output_tokens} tokens")

    # 同一 batch 的 16 个请求一起结束：按请求展开后求分位，对齐 vllm 的逐请求口径。
    lats = sorted(lat for lat in per_request_latencies for _ in range(batch_size))
    print("\n=== 压测结果 ===")
    print(f"  成功: {total_requests}/{total_requests}  总耗时: {total_elapsed:.2f}s")
    if lats:
        print(f"  延迟  avg={sum(lats) / len(lats):.2f}s  "
              f"p50={_pct(lats, 0.50):.2f}s  p90={_pct(lats, 0.90):.2f}s")
    if total_elapsed > 0:
        print(f"  吞吐: {total_out / total_elapsed:.1f} output tokens/s")

    # ── Prefill ──
    prefill_s = sum(meter.prefill_host_ms) / 1000.0
    prefill_tokens = batch_size * prompt_tokens * num_batches
    if prefill_s > 0:
        print(f"\n[Prefill] {num_batches} 次 (每次 batch={batch_size}): "
              f"{prefill_tokens} tokens  {prefill_s * 1000:.2f} ms  "
              f"({prefill_tokens / prefill_s:.0f} prefill tokens/s)")

    # ── Decode 关键指标（步周期 = 真实单步；device wall 来自 L3 worker timing）──
    periods = meter.decode_periods_ms
    if not periods:
        return
    print("\n" + "=" * 64)
    print("=== Decode 真实耗时（来自 L3 worker device timing）===")

    decode_wall = sum(periods) / 1000.0
    print(f"  校验: 预填+解码 {prefill_s + decode_wall:.2f}s  ←→  压测墙钟 {total_elapsed:.2f}s  "
          f"(差 {abs(total_elapsed - prefill_s - decode_wall):.2f}s)")

    sp = sorted(periods)
    mean_p = sum(periods) / len(periods)
    print(f"  decode 步数 {len(periods)}   平均 batch {batch_size:.1f}/{batch_size}")
    print("  ── 关键指标 ──")
    print(f"  TPOT(真实单步):  {mean_p:6.2f} ms  (p50 {_pct(sp, 0.50):.2f}, p99 {_pct(sp, 0.99):.2f})")
    dev = meter.decode_device_ms
    if dev:
        dev_mean = sum(dev) / len(dev)
        occ = dev_mean / mean_p * 100 if mean_p else 0.0
        print(f"  device 计算:     {dev_mean:6.2f} ms  (NPU 占用 {occ:.1f}%)")
        print(f"  host 间隙:       {mean_p - dev_mean:6.2f} ms")
    print(f"  整 batch 吞吐:   {batch_size * 1000 / mean_p:6.1f} tok/s   单请求 {1000 / mean_p:.1f} tok/s")


def main() -> None:
    args = build_parser().parse_args()
    get_profiler(process_name="npu_stress")

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    prompt = Path(args.prompt_file).read_text().strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt_file}")
    if args.batch_size > args.max_batch_size:
        raise ValueError(
            f"--batch-size {args.batch_size} exceeds --max-batch-size {args.max_batch_size}"
        )

    profile_enabled = args.profile or args.profile_verbose
    collector = _TimingCollector() if profile_enabled else None

    if args.scope_stats_dir is not None:
        _enable_l3_scope_stats(Path(args.scope_stats_dir).resolve())

    kv_cache_manager = KvCacheManager()
    executor = PyptoExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_id=args.device_id,
        save_kernels_dir=args.save_kernels_dir,
        l3_trace=args.profile_verbose,
        prefill_on_cpu=args.prefill_on_cpu,
        cpu_prefill_cache_dir=args.prefill_cache_dir,
        dump_kv=args.dump_kv is not None,
    )
    engine = LLMEngine(kv_cache_manager=kv_cache_manager, executor=executor)

    if args.num_layers_override is not None:
        _install_num_layers_override(engine, args.num_layers_override)
        print(f"[override] num_hidden_layers -> {args.num_layers_override}", flush=True)

    try:
        init_t0 = time.perf_counter()
        engine.init_model(
            model_id=args.model_id,
            model_dir=str(model_dir),
            model_format="huggingface",
            runtime_config=RuntimeConfig(
                page_size=128,
                max_batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len,
                max_new_tokens=args.max_new_tokens,
                device="cpu",
                kv_dtype="bfloat16",
                weight_dtype="float32",
                prefill_chunk_size=args.prefill_chunk_size,
            ),
        )
        if collector is not None:
            collector.phases["init_model"] = time.perf_counter() - init_t0
            InstallProfiling(engine, args.model_id, collector)

        config = GenerateConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=False,
        )

        meter = InstallStepMeter(engine, args.model_id)
        if args.dump_kv:
            _install_kv_dump(engine, args.model_id, args.dump_kv)
        prompts = [prompt] * args.batch_size

        # Report the real tokenized prompt length rather than the ~3.3k estimate.
        record = engine._models[args.model_id]  # type: ignore[attr-defined]
        prompt_tokens = len(record.tokenizer.encode(prompt))

        per_request_latencies: list[float] = []
        output_tokens = args.max_new_tokens
        run_t0 = time.perf_counter()
        with profile_span(
            "npu_stress.run",
            cat="request",
            args={"batch_size": args.batch_size, "num_batches": args.num_batches},
        ):
            for batch_idx in range(args.num_batches):
                batch_t0 = time.perf_counter()
                results = engine.generate_batch(args.model_id, prompts, config)
                per_request_latencies.append(time.perf_counter() - batch_t0)
                output_tokens = max(len(r.token_ids) for r in results)
        total_elapsed = time.perf_counter() - run_t0

        # Generated tokens — for the CPU-vs-NPU prefill A/B correctness check
        # (greedy temperature=0 is deterministic, so identical token ids => the
        # CPU-prefill KV matches the NPU prefill kernel's). Print row 0 only.
        print("\n=== 生成结果 (row 0) ===", flush=True)
        print(f"  token_ids: {results[0].token_ids}", flush=True)
        print(f"  text: {results[0].text!r}", flush=True)

        _print_report(
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            per_request_latencies=per_request_latencies,
            total_elapsed=total_elapsed,
            meter=meter,
        )

        if collector is not None:
            collector.phases["generate_total"] = total_elapsed
            PrintTimingReport(
                collector,
                num_tokens=args.batch_size * output_tokens * args.num_batches,
                verbose=args.profile_verbose,
            )
    finally:
        try:
            executor.close()
        finally:
            merge_profile()


if __name__ == "__main__":
    main()
