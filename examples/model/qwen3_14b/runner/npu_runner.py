# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from python.core.model_runner import ModelRunner
from python.core.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from python.profile import profile_span


def _kernel_trace_name(kernel_name: str) -> str:
    if "prefill" in kernel_name:
        return "kernel.prefill_fwd"
    if "decode" in kernel_name:
        return "kernel.decode_fwd"
    return f"kernel.{kernel_name}"


def _run_timing_us(timing: Any) -> tuple[float | None, float | None]:
    if timing is None:
        return None, None
    host_wall_us = getattr(timing, "host_wall_us", None)
    device_wall_us = getattr(timing, "device_wall_us", None)
    if host_wall_us is not None:
        host_wall_us = float(host_wall_us)
    if device_wall_us is not None:
        device_wall_us = float(device_wall_us)
    return host_wall_us, device_wall_us


def _add_run_timing_args(args: dict[str, Any], timing: Any) -> None:
    host_wall_us, device_wall_us = _run_timing_us(timing)
    if host_wall_us is not None:
        args["host_wall_us"] = host_wall_us
        args["host_wall_ms"] = host_wall_us / 1000.0
    if device_wall_us is not None:
        args["device_wall_us"] = device_wall_us
        args["device_wall_ms"] = device_wall_us / 1000.0


@dataclass
class _L3Callable:
    """HOST-dispatched compiled program and launch metadata."""

    compiled: object
    name: str
    block_dim: int
    aicpu_thread_num: int
    dispatch_args: tuple[Any, ...] = ()


@dataclass
class _CompiledKernels:
    """Compiled Qwen3-14B kernels and immutable runtime tensors."""

    prefill: _L3Callable
    decode: _L3Callable
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    decode_weights: dict[str, torch.Tensor]
    prefill_hidden_buffer: torch.Tensor
    prefill_seq_lens_buffer: torch.Tensor
    prefill_chunk_lens_buffer: torch.Tensor
    prefill_chunk_offsets_buffer: torch.Tensor
    prefill_block_table_buffer: torch.Tensor
    prefill_slot_mapping_buffer: torch.Tensor
    prefill_logits_buffer: torch.Tensor
    decode_hidden_buffer: torch.Tensor
    decode_seq_lens_buffer: torch.Tensor
    decode_block_table_buffer: torch.Tensor
    decode_slot_mapping_buffer: torch.Tensor
    decode_logits_buffer: torch.Tensor
    # Torch reference prefill (pypto-lib prefill_fwd.golden_qwen3_14b_prefill). Used
    # only by the optional CPU-prefill path; returns the post-chunk (k_cache, v_cache)
    # in the kernel's paged BF16 layout and fills the logits buffer in-place.
    cpu_prefill_golden: Any = None
    # Pre-fork SHARED-memory staging buffer for host->device KV writes. The L3 chip
    # worker is forked and can only read host memory inherited at fork, so a freshly
    # allocated source buffer is invisible to it (copy_to -> 107017). KV pages are
    # staged through this buffer (sized to one row's largest coalesced page run) and
    # then copy_to'd. None unless prefill_on_cpu was set at compile time.
    cpu_prefill_stage: torch.Tensor = None


@dataclass
class _PrefillInputs:
    """Host tensors passed to the prefill kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    chunk_lens: torch.Tensor
    chunk_offsets: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeInputs:
    """Active user rows prepared for decode."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeKernelInputs:
    """Fixed-batch tensors passed to the fused decode kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    logits: torch.Tensor


@dataclass
class _StaticDeviceTensor:
    """A shared host tensor to upload into the shared L3 worker once."""

    tensor: torch.Tensor


@dataclass
class _StaticKernelArgs:
    """Static worker-resident kernel arguments reused across dispatches."""

    final_norm_weight: _StaticDeviceTensor
    rope_cos: _StaticDeviceTensor
    rope_sin: _StaticDeviceTensor
    padded_lm_head_weight: _StaticDeviceTensor
    decode_weights: dict[str, _StaticDeviceTensor]


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        compiled: _CompiledKernels,
        prefill_on_cpu: bool = False,
        cpu_prefill_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[tuple[int, tuple[int, ...], torch.dtype], object] = {}
        self._static_args: _StaticKernelArgs | None = None
        self._pending_kv_cache_specs: dict[str, tuple[ModelConfig, RuntimeConfig]] = {}
        # CPU-prefill path: compute the prompt KV (+ first-token logits) on host with
        # the torch reference, then push the KV into the device pool the decode kernel
        # reads. Sidesteps the NPU prefill kernel's ring-arena OOM at 40 layers / long
        # context. Env override allows toggling without re-plumbing executor_kwargs.
        self._prefill_on_cpu = prefill_on_cpu or os.environ.get(
            "PYPTO_QWEN3_PREFILL_ON_CPU", ""
        ).lower() in ("1", "true", "yes")
        # Per-model host KV shadow ([cache_rows, head_dim] BF16, full pool shape) so
        # chunked CPU prefill accumulates prior windows' KV before the device push.
        self._cpu_kv_shadow: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Optional on-disk reuse of CPU-prefill results: caches the LOGICAL
        # (position-major, page-independent) KV + first-token logits per unique
        # prompt, so a rerun skips the minutes-long host loop. Keyed by token ids +
        # model dims, so it is robust to a different physical page assignment.
        cache_dir = cpu_prefill_cache_dir or os.environ.get("PYPTO_QWEN3_PREFILL_CACHE_DIR")
        self._cpu_prefill_cache_dir = Path(cache_dir) if cache_dir else None
        if compiled is not None:
            self._share_static_kernel_tensors()
            self._static_args = self._build_static_kernel_args()

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        """Create the L3 worker-resident cache before the first request."""
        if model_id in self._kv_caches:
            return
        self._pending_kv_cache_specs[model_id] = (config, runtime)
        with profile_span("Qwen314BModelRunner.prepare_l3_worker", cat="executor"):
            self._shared_l3_worker()
        with profile_span("Qwen314BModelRunner.upload_static_tensors", cat="executor"):
            self._materialize_static_tensors()
        with profile_span("Qwen314BModelRunner.init_kv_cache", cat="executor"):
            ModelRunner.init_kv_cache(self, model_id, config, runtime)

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        """Allocate one worker-resident KV cache tensor shared by prefill/decode."""
        return self._shared_l3_worker().alloc_tensor(shape, dtype)

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        """Release one worker-resident KV cache tensor."""
        worker = self._l3_worker
        if worker is not None:
            worker.free_tensor(tensor)

    def _materialize_kv_cache(self, model: RuntimeModel) -> Any:
        """Return the worker-resident KV cache, allocating only as a fallback."""
        kv_cache = self._kv_caches.get(model.config.model_id)
        if kv_cache is not None:
            return kv_cache
        spec = self._pending_kv_cache_specs.get(model.config.model_id)
        if spec is None:
            spec = (model.config, model.runtime)
            self._pending_kv_cache_specs[model.config.model_id] = spec
        ModelRunner.init_kv_cache(self, model.config.model_id, spec[0], spec[1])
        return self._kv_caches[model.config.model_id]

    @staticmethod
    def _validate_kv_cache_bounds(
        model: RuntimeModel,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache: Any,
    ) -> None:
        """Fail on host before an invalid KV page id reaches the NPU kernel."""
        valid_blocks = block_table[block_table >= 0]
        valid_slots = slot_mapping[slot_mapping >= 0]
        if valid_blocks.numel() == 0 and valid_slots.numel() == 0:
            return
        max_block_id = int(valid_blocks.max().item()) if valid_blocks.numel() else -1
        max_slot_block = int(valid_slots.max().item()) // model.runtime.page_size if valid_slots.numel() else -1
        max_page_id = max(max_block_id, max_slot_block)
        rows_per_layer = cache.shape[0] // model.config.num_hidden_layers
        max_pages = rows_per_layer // (model.config.num_key_value_heads * model.runtime.page_size)
        if max_page_id >= max_pages:
            raise RuntimeError(
                "KV cache page id exceeds runner device cache capacity: "
                f"max_page_id={max_page_id}, max_pages={max_pages}, "
                f"cache_shape={cache.shape}, block_table_shape={tuple(block_table.shape)}, "
                f"slot_mapping_shape={tuple(slot_mapping.shape)}"
            )

    def _share_static_kernel_tensors(self) -> None:
        """Move static kernel inputs to shared memory before worker creation."""
        for tensor in self._iter_static_host_tensors():
            self._share_cpu_tensor(tensor)

    def _iter_static_host_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return host tensors that must be shared before the worker forks."""
        compiled = self._compiled
        return (
            compiled.final_norm_weight,
            compiled.rope_cos,
            compiled.rope_sin,
            compiled.padded_lm_head_weight,
            *compiled.decode_weights.values(),
            compiled.prefill_hidden_buffer,
            compiled.prefill_seq_lens_buffer,
            compiled.prefill_chunk_lens_buffer,
            compiled.prefill_chunk_offsets_buffer,
            compiled.prefill_block_table_buffer,
            compiled.prefill_slot_mapping_buffer,
            compiled.prefill_logits_buffer,
            compiled.decode_hidden_buffer,
            compiled.decode_seq_lens_buffer,
            compiled.decode_block_table_buffer,
            compiled.decode_slot_mapping_buffer,
            compiled.decode_logits_buffer,
        )

    def _build_static_kernel_args(self) -> _StaticKernelArgs:
        """Create static device-upload markers once per runner."""
        compiled = self._compiled
        return _StaticKernelArgs(
            final_norm_weight=self._static_device_tensor(compiled.final_norm_weight),
            rope_cos=self._static_device_tensor(compiled.rope_cos),
            rope_sin=self._static_device_tensor(compiled.rope_sin),
            padded_lm_head_weight=self._static_device_tensor(compiled.padded_lm_head_weight),
            decode_weights={
                name: self._static_device_tensor(tensor)
                for name, tensor in compiled.decode_weights.items()
            },
        )

    def _require_static_args(self) -> _StaticKernelArgs:
        """Return prebuilt static args for dispatch."""
        if self._static_args is None:
            raise RuntimeError("Qwen314BModelRunner static kernel args are not initialized")
        return self._static_args

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run all-layer prefill and return next-token logits.

        Dispatches to the NPU prefill kernel by default, or to the host torch
        reference when ``prefill_on_cpu`` is set (writes the resulting KV into the
        same device pool the decode kernel reads, so decode is unchanged).
        """
        if self._prefill_on_cpu:
            return self._run_prefill_cpu(model, batch)
        compiled = self._compiled
        prefill_inputs = self._prepare_prefill_inputs(model, batch)

        logits_padded = compiled.prefill_logits_buffer

        kv_cache = self._materialize_kv_cache(model)
        k_cache = kv_cache.key_pages
        v_cache = kv_cache.value_pages
        self._validate_kv_cache_bounds(model, prefill_inputs.block_table, prefill_inputs.slot_mapping, k_cache)

        self._run_distributed_program(
            compiled.prefill,
            *self._prefill_kernel_args(prefill_inputs, k_cache, v_cache, logits_padded),
        )

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[: prefill_inputs.actual_batch, : model.config.vocab_size],
        )

    def _run_prefill_cpu(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Compute prefill on host and push the prompt KV into the device pool.

        Reuses ``prefill_fwd.golden_qwen3_14b_prefill`` (the torch reference that
        mirrors the NPU prefill kernel's precision path) over the SAME packed
        inputs, weights and RoPE tables the kernel would use. The reference returns
        the post-chunk paged KV (BF16, identical row layout to the kernel), which we
        copy into ``key_pages`` / ``value_pages`` — the device-resident pool the
        decode kernel reads via ``block_table`` / ``slot_mapping``. No NPU prefill
        dispatch, so the 40-layer ring-arena OOM never arises.
        """
        compiled = self._compiled
        if compiled.cpu_prefill_golden is None:
            raise RuntimeError(
                "prefill_on_cpu is set but no cpu_prefill_golden was compiled in; "
                "the executor must pass prefill_fwd.golden_qwen3_14b_prefill."
            )
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
        kv_cache = self._materialize_kv_cache(model)
        k_dev = kv_cache.key_pages
        v_dev = kv_cache.value_pages
        self._validate_kv_cache_bounds(model, prefill_inputs.block_table, prefill_inputs.slot_mapping, k_dev)

        out_buf = compiled.prefill_logits_buffer
        if self._cpu_prefill_load_all(model, batch, prefill_inputs, k_dev, v_dev, out_buf):
            self._mark_prefill_tokens_used(batch)
            return PrefillResult(
                last_hidden=None,
                logits=out_buf[: prefill_inputs.actual_batch, : model.config.vocab_size],
            )

        k_host, v_host = self._get_cpu_kv_shadow(model.config.model_id, k_dev, v_dev)
        weights = compiled.decode_weights
        tensors = {
            "hidden_states": prefill_inputs.hidden,
            "seq_lens": prefill_inputs.seq_lens,
            "chunk_lens": prefill_inputs.chunk_lens,
            "chunk_offsets": prefill_inputs.chunk_offsets,
            "input_rms_weight": weights["decode_input_rms_weight"],
            "wq": weights["decode_wq"],
            "wk": weights["decode_wk"],
            "wv": weights["decode_wv"],
            "q_norm_weight": weights["decode_q_norm_weight"],
            "k_norm_weight": weights["decode_k_norm_weight"],
            "rope_cos": compiled.rope_cos,
            "rope_sin": compiled.rope_sin,
            "block_table": prefill_inputs.block_table,
            "slot_mapping": prefill_inputs.slot_mapping,
            "k_cache": k_host,
            "v_cache": v_host,
            "wo": weights["decode_wo"],
            "post_rms_weight": weights["decode_post_rms_weight"],
            "w_gate": weights["decode_w_gate"],
            "w_up": weights["decode_w_up"],
            "w_down": weights["decode_w_down"],
            "final_norm_weight": compiled.final_norm_weight,
            "lm_head_weight": compiled.padded_lm_head_weight,
            "out": out_buf,
        }

        with profile_span("Qwen314BModelRunner.cpu_prefill_golden", cat="executor"):
            # The reference clones k_cache/v_cache internally, so the shadow is not
            # mutated in place; rebind it to the returned (accumulated) caches so a
            # later chunk for the same prompt attends over prior windows' KV.
            k_new, v_new = compiled.cpu_prefill_golden(
                tensors, progress=self._make_cpu_prefill_progress()
            )
        self._cpu_kv_shadow[model.config.model_id] = (k_new, v_new)

        with profile_span("Qwen314BModelRunner.cpu_prefill_kv_push", cat="executor"):
            self._push_kv_pages_to_device(model, batch, prefill_inputs, k_new, v_new, k_dev, v_dev)

        self._cpu_prefill_save_all(model, batch, prefill_inputs, k_new, v_new, out_buf)
        self._mark_prefill_tokens_used(batch)
        return PrefillResult(
            last_hidden=None,
            logits=out_buf[: prefill_inputs.actual_batch, : model.config.vocab_size],
        )

    @staticmethod
    def _mark_prefill_tokens_used(batch: PrefillBatch) -> None:
        """Advance each allocation's used-token count to its post-prefill length."""
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)

    @staticmethod
    def _row_page_ids(batch: PrefillBatch, batch_idx: int) -> list[int]:
        """Return the KV page ids backing one batch row (alloc or explicit blocks)."""
        if batch_idx < len(batch.kv_allocations) and batch.kv_allocations[batch_idx] is not None:
            return batch.kv_allocations[batch_idx].page_ids
        if batch_idx < len(batch.block_ids):
            return batch.block_ids[batch_idx]
        return []

    def _cpu_prefill_cache_key(self, model: RuntimeModel, token_ids_row: torch.Tensor, seq_len: int) -> str:
        """Content hash identifying a prompt's CPU-prefill result.

        Covers the prompt tokens plus every model dim that changes the KV/logits,
        so a stale cache from a different model / shape can never be mis-applied.
        """
        cfg = model.config
        header = (
            f"{cfg.model_id}|{seq_len}|{cfg.num_hidden_layers}|{cfg.num_key_value_heads}|"
            f"{cfg.head_dim}|{cfg.hidden_size}|{getattr(cfg, 'rope_theta', 0.0)}|"
            f"{self._compiled.padded_vocab}"
        )
        h = hashlib.sha1(header.encode())
        h.update(token_ids_row[:seq_len].to(torch.int64).cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _cpu_prefill_load_all(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
        prefill_inputs: _PrefillInputs,
        k_dev: DeviceTensor,
        v_dev: DeviceTensor,
        out_buf: torch.Tensor,
    ) -> bool:
        """Serve the whole batch from disk if every row is a cache hit.

        Only full-prompt rows (chunk == whole sequence) are cacheable, since the
        saved logical KV must cover the full prefix [0, seq_len). Returns True iff
        all active rows were loaded (KV scattered to device, logits filled).
        """
        if self._cpu_prefill_cache_dir is None:
            return False
        entries = []
        for batch_idx in range(prefill_inputs.actual_batch):
            seq_len = int(prefill_inputs.seq_lens[batch_idx].item())
            chunk_len = int(prefill_inputs.chunk_lens[batch_idx].item())
            if chunk_len != seq_len:
                return False
            key = self._cpu_prefill_cache_key(model, batch.token_ids[batch_idx], seq_len)
            path = self._cpu_prefill_cache_dir / f"{key}.pt"
            if not path.is_file():
                return False
            entries.append((batch_idx, seq_len, path))

        # Rebuild each row's pages in the host pool from the logical cache, then
        # copy just those pages to the device (same coalesced fast path as a fresh
        # compute's push — only the request's own pages are touched).
        k_host, v_host = self._get_cpu_kv_shadow(model.config.model_id, k_dev, v_dev)
        worker = self._shared_l3_worker()
        vocab = model.config.vocab_size
        for batch_idx, seq_len, path in entries:
            data = torch.load(path, map_location="cpu")
            out_buf[batch_idx, :vocab] = data["logits"][:vocab]
            page_ids = self._row_page_ids(batch, batch_idx)
            self._scatter_logical_to_host(k_host, data["k"], page_ids, seq_len, model)
            self._scatter_logical_to_host(v_host, data["v"], page_ids, seq_len, model)
            self._copy_row_pages(worker, k_host, k_dev, page_ids, seq_len, model)
            self._copy_row_pages(worker, v_host, v_dev, page_ids, seq_len, model)
        print(
            f"[cpu-prefill] cache HIT: {len(entries)} row(s) loaded from "
            f"{self._cpu_prefill_cache_dir} (skipped host prefill)",
            flush=True,
        )
        return True

    def _cpu_prefill_save_all(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
        prefill_inputs: _PrefillInputs,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> None:
        """Persist each full-prompt row's logical KV + logits for later reuse."""
        if self._cpu_prefill_cache_dir is None:
            return
        self._cpu_prefill_cache_dir.mkdir(parents=True, exist_ok=True)
        vocab = model.config.vocab_size
        saved = 0
        for batch_idx in range(prefill_inputs.actual_batch):
            seq_len = int(prefill_inputs.seq_lens[batch_idx].item())
            chunk_len = int(prefill_inputs.chunk_lens[batch_idx].item())
            if chunk_len != seq_len:
                continue
            key = self._cpu_prefill_cache_key(model, batch.token_ids[batch_idx], seq_len)
            path = self._cpu_prefill_cache_dir / f"{key}.pt"
            if path.is_file():
                continue
            page_ids = self._row_page_ids(batch, batch_idx)
            payload = {
                "seq_len": seq_len,
                "k": self._gather_logical_from_pool(k_pool, page_ids, seq_len, model),
                "v": self._gather_logical_from_pool(v_pool, page_ids, seq_len, model),
                "logits": out_buf[batch_idx, :vocab].clone(),
            }
            tmp = path.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.rename(path)  # atomic publish; a killed run never leaves a half file
            saved += 1
        if saved:
            print(
                f"[cpu-prefill] cache SAVE: {saved} new row(s) -> {self._cpu_prefill_cache_dir}",
                flush=True,
            )

    def _gather_logical_from_pool(
        self,
        pool: torch.Tensor,
        page_ids: list[int],
        seq_len: int,
        model: RuntimeModel,
    ) -> torch.Tensor:
        """Gather a row's paged KV into a [num_layers, seq_len, num_kv_heads, head_dim] tensor."""
        page_size = model.runtime.page_size
        num_kv_heads = model.config.num_key_value_heads
        num_layers = model.config.num_hidden_layers
        head_dim = model.config.head_dim
        rows_per_layer = pool.shape[0] // num_layers
        page_rows = num_kv_heads * page_size
        out = torch.zeros((num_layers, seq_len, num_kv_heads, head_dim), dtype=pool.dtype)
        for layer_idx in range(num_layers):
            base = layer_idx * rows_per_layer
            for page_pos, phys_page in enumerate(page_ids):
                p0 = page_pos * page_size
                if p0 >= seq_len:
                    break
                n = min(page_size, seq_len - p0)
                block = pool[base + phys_page * page_rows : base + phys_page * page_rows + page_rows]
                block = block.view(num_kv_heads, page_size, head_dim)
                out[layer_idx, p0 : p0 + n] = block[:, :n, :].transpose(0, 1)
        return out

    def _scatter_logical_to_host(
        self,
        pool: torch.Tensor,
        log: torch.Tensor,
        page_ids: list[int],
        seq_len: int,
        model: RuntimeModel,
    ) -> None:
        """Scatter logical KV ([num_layers, seq_len, num_kv_heads, head_dim]) into a
        host paged pool in place (inverse of ``_gather_logical_from_pool``).

        Writes directly into the full host pool so the caller can bulk-upload it;
        no per-page device copies.
        """
        page_size = model.runtime.page_size
        num_kv_heads = model.config.num_key_value_heads
        num_layers = model.config.num_hidden_layers
        head_dim = model.config.head_dim
        rows_per_layer = pool.shape[0] // num_layers
        page_rows = num_kv_heads * page_size
        for layer_idx in range(num_layers):
            base = layer_idx * rows_per_layer
            for page_pos, phys_page in enumerate(page_ids):
                p0 = page_pos * page_size
                if p0 >= seq_len:
                    break
                n = min(page_size, seq_len - p0)
                block = pool[base + phys_page * page_rows : base + phys_page * page_rows + page_rows]
                block = block.view(num_kv_heads, page_size, head_dim)
                block[:, :n, :] = log[layer_idx, p0 : p0 + n].transpose(0, 1)

    @staticmethod
    def _make_cpu_prefill_progress():
        """Return a throttled ``progress(done, total)`` callback for host prefill.

        Emitted as plain log lines (not a TTY carriage-return bar) so it stays
        readable in a redirected stress log: at most one line every few seconds,
        plus a guaranteed final 100% line. ETA assumes roughly uniform per-step
        cost (each step is one transformer layer over one batch row).
        """
        state = {"t0": time.perf_counter(), "last": -1.0}

        def progress(done: int, total: int) -> None:
            now = time.perf_counter()
            if done < total and now - state["last"] < 5.0:
                return
            state["last"] = now
            elapsed = now - state["t0"]
            frac = (done / total) if total else 1.0
            eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
            print(
                f"[cpu-prefill] {done}/{total} layer-rows ({frac * 100:.0f}%) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

        return progress

    def _get_cpu_kv_shadow(
        self,
        model_id: str,
        k_dev: DeviceTensor,
        v_dev: DeviceTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lazily allocate) the host KV shadow matching the device pool.

        Full-pool shape so the reference can index by absolute physical row exactly
        as the kernel does. Allocated once per model and reused across requests;
        stale rows from freed pages are never read (each request only attends over
        positions it has written).
        """
        shadow = self._cpu_kv_shadow.get(model_id)
        if shadow is None:
            shape = tuple(k_dev.shape)
            shadow = (
                torch.zeros(shape, dtype=k_dev.dtype),
                torch.zeros(tuple(v_dev.shape), dtype=v_dev.dtype),
            )
            self._cpu_kv_shadow[model_id] = shadow
        return shadow

    def _push_kv_pages_to_device(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
        prefill_inputs: _PrefillInputs,
        k_host: torch.Tensor,
        v_host: torch.Tensor,
        k_dev: DeviceTensor,
        v_dev: DeviceTensor,
    ) -> None:
        """Copy each request's KV pages from the host pool into the device pool.

        Touches ONLY the rows backing this batch's pages (never clobbers another
        request's live device KV). To avoid the tens-of-thousands of tiny
        per-(page, layer) orchestrator round-trips that made this take hours, runs
        of physically-consecutive pages are coalesced into one large ``copy_to``
        per layer — so a contiguous allocation is just ``num_layers`` copies/row.
        """
        worker = self._shared_l3_worker()
        for batch_idx in range(prefill_inputs.actual_batch):
            if int(prefill_inputs.chunk_lens[batch_idx].item()) <= 0:
                continue
            seq_len = int(prefill_inputs.seq_lens[batch_idx].item())
            page_ids = self._row_page_ids(batch, batch_idx)
            self._copy_row_pages(worker, k_host, k_dev, page_ids, seq_len, model)
            self._copy_row_pages(worker, v_host, v_dev, page_ids, seq_len, model)

    def dump_request_kv(self, model: RuntimeModel, batch: PrefillBatch, path: str) -> None:
        """Debug: read each request's prompt KV pages back from the device pool and
        save logical [num_layers, seq_len, num_kv_heads, head_dim] K/V + token ids.

        Call right after prefill (before decode appends new-token KV) to capture the
        pure prompt KV. Used to compare CPU-prefill vs NPU-prefill KV numerically;
        reads from the device in both cases so it is apples-to-apples.
        """
        kv = self._kv_caches.get(model.config.model_id)
        if kv is None:
            raise RuntimeError("dump_request_kv: KV cache not initialized")
        stage = self._compiled.cpu_prefill_stage
        if stage is None:
            raise RuntimeError("dump_request_kv requires the pre-fork staging buffer (pass dump_kv at compile)")
        k_dev, v_dev = kv.key_pages, kv.value_pages
        worker = self._shared_l3_worker()
        k_host = torch.zeros(tuple(k_dev.shape), dtype=k_dev.dtype)
        v_host = torch.zeros(tuple(v_dev.shape), dtype=v_dev.dtype)
        actual = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        payloads = []
        for batch_idx in range(actual):
            seq_len = int(batch.seq_lens[batch_idx].item())
            page_ids = self._row_page_ids(batch, batch_idx)
            self._read_pages_from_device(worker, k_dev, k_host, stage, page_ids, seq_len, model)
            self._read_pages_from_device(worker, v_dev, v_host, stage, page_ids, seq_len, model)
            payloads.append({
                "seq_len": seq_len,
                "token_ids": batch.token_ids[batch_idx][:seq_len].to(torch.int64).cpu().tolist(),
                "k": self._gather_logical_from_pool(k_host, page_ids, seq_len, model),
                "v": self._gather_logical_from_pool(v_host, page_ids, seq_len, model),
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payloads, path)

    def _read_pages_from_device(
        self,
        worker: Any,
        dev: DeviceTensor,
        host_pool: torch.Tensor,
        stage: torch.Tensor,
        page_ids: list[int],
        seq_len: int,
        model: RuntimeModel,
    ) -> None:
        """Read one request's [0, seq_len) KV pages (coalesced) from the device pool
        into ``host_pool`` via the shared staging buffer (inverse of _copy_row_pages)."""
        page_size = model.runtime.page_size
        num_kv_heads = model.config.num_key_value_heads
        num_layers = model.config.num_hidden_layers
        rows_per_layer = dev.shape[0] // num_layers
        page_rows = num_kv_heads * page_size
        row_bytes = dev.shape[1] * torch.tensor([], dtype=dev.dtype).element_size()
        num_pages = (seq_len + page_size - 1) // page_size
        runs = self._coalesce_pages(page_ids, num_pages)
        for layer_idx in range(num_layers):
            base = layer_idx * rows_per_layer
            for first_page, count in runs:
                r0 = base + first_page * page_rows
                nrows = count * page_rows
                worker.copy_from(stage[:nrows].data_ptr(), dev.data_ptr + r0 * row_bytes, nrows * row_bytes)
                host_pool[r0 : r0 + nrows].copy_(stage[:nrows])

    @staticmethod
    def _coalesce_pages(page_ids: list[int], num_pages: int) -> list[tuple[int, int]]:
        """Group the first ``num_pages`` physical pages into (first_page, count) runs."""
        pages = list(page_ids[:num_pages])
        runs: list[tuple[int, int]] = []
        i = 0
        while i < len(pages):
            j = i
            while j + 1 < len(pages) and pages[j + 1] == pages[j] + 1:
                j += 1
            runs.append((pages[i], j - i + 1))
            i = j + 1
        return runs

    def _copy_row_pages(
        self,
        worker: Any,
        host_pool: torch.Tensor,
        dev: DeviceTensor,
        page_ids: list[int],
        seq_len: int,
        model: RuntimeModel,
    ) -> None:
        """Copy one request's [0, seq_len) KV pages (coalesced) into the device pool.

        Each run is staged through the pre-fork shared buffer before copy_to: the
        forked chip worker can only read host memory inherited at fork, so the
        copy_to source MUST be that shared buffer, not a freshly sliced tensor.
        """
        stage = self._compiled.cpu_prefill_stage
        if stage is None:
            raise RuntimeError(
                "cpu_prefill_stage is not allocated; prefill_on_cpu must be set when "
                "the executor compiles the model (it allocates the pre-fork shared buffer)."
            )
        page_size = model.runtime.page_size
        num_kv_heads = model.config.num_key_value_heads
        num_layers = model.config.num_hidden_layers
        rows_per_layer = dev.shape[0] // num_layers
        page_rows = num_kv_heads * page_size
        row_bytes = dev.shape[1] * torch.tensor([], dtype=dev.dtype).element_size()
        num_pages = (seq_len + page_size - 1) // page_size
        runs = self._coalesce_pages(page_ids, num_pages)
        for layer_idx in range(num_layers):
            base = layer_idx * rows_per_layer
            for first_page, count in runs:
                r0 = base + first_page * page_rows
                nrows = count * page_rows
                stage[:nrows].copy_(host_pool[r0 : r0 + nrows])
                worker.copy_to(dev.data_ptr + r0 * row_bytes, stage[:nrows].data_ptr(), nrows * row_bytes)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the fused all-layer PAGED ``decode_layer.decode_fwd`` and return logits.

        ``decode_fwd`` runs all NUM_LAYERS + the LM head in one dispatch over the
        PAGED KV pool, addressing KV via ``block_table`` + ``slot_mapping`` — the
        SAME device-resident KV pool prefill writes (``self._kv_caches``), so prompt
        KV is already in place with no bridge. KV is keyed by block_table page id, not by
        kernel row, so a request may occupy any row each step (no stable-slot shim).

        The kernel is FIXED-BATCH (it computes all max_batch_size rows and writes
        each row's current-token KV). Pad the active batch up to the kernel batch by
        REPLICATING active row 0's inputs into the padding rows: those rows then
        recompute row 0's K/V and write row 0's own slot with byte-identical values
        (an idempotent, safe write), and their logits are trimmed off below. This
        avoids padded rows clobbering an unrelated request's physical page.
        """
        compiled = self._compiled
        model_id = model.config.model_id
        decode_inputs = self._prepare_decode_inputs(model, batch)

        kv_cache = self._kv_caches.get(model_id)
        if kv_cache is None:
            raise RuntimeError(f"KV cache for model {model_id!r} is not initialized")
        k_cache = kv_cache.key_pages
        v_cache = kv_cache.value_pages

        kernel_inputs = self._pad_decode_inputs(model, decode_inputs)

        # Padded block_table / slot_mapping only ever reference row 0's
        # already-valid pages, so bound-check exactly what the kernel will read.
        self._validate_kv_cache_bounds(model, kernel_inputs.block_table, kernel_inputs.slot_mapping, k_cache)

        self._run_distributed_program(
            compiled.decode,
            *self._decode_kernel_args(kernel_inputs, k_cache, v_cache),
        )
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(
            hidden_states=decode_inputs.hidden.float(),
            logits=kernel_inputs.logits[: kernel_inputs.actual_batch, : model.config.vocab_size].to(
                decode_inputs.hidden.device
            ),
        )

    def _prefill_kernel_args(
        self,
        inputs: _PrefillInputs,
        k_cache: DeviceTensor,
        v_cache: DeviceTensor,
        logits: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_prefill_host`` signature order."""
        static = self._require_static_args()
        weights = static.decode_weights
        return (
            inputs.hidden,
            inputs.seq_lens,
            inputs.chunk_lens,
            inputs.chunk_offsets,
            weights["decode_input_rms_weight"],
            weights["decode_wq"],
            weights["decode_wk"],
            weights["decode_wv"],
            weights["decode_q_norm_weight"],
            weights["decode_k_norm_weight"],
            static.rope_cos,
            static.rope_sin,
            inputs.block_table,
            inputs.slot_mapping,
            k_cache,
            v_cache,
            weights["decode_wo"],
            weights["decode_post_rms_weight"],
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            logits,
        )

    def _decode_kernel_args(
        self,
        inputs: _DecodeKernelInputs,
        k_cache: DeviceTensor,
        v_cache: DeviceTensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_decode_host`` signature order."""
        static = self._require_static_args()
        weights = static.decode_weights
        return (
            inputs.hidden,
            weights["decode_input_rms_weight"],
            weights["decode_wq"],
            weights["decode_wk"],
            weights["decode_wv"],
            weights["decode_q_norm_weight"],
            weights["decode_k_norm_weight"],
            inputs.seq_lens,
            inputs.block_table,
            inputs.slot_mapping,
            static.rope_cos,
            static.rope_sin,
            k_cache,
            v_cache,
            weights["decode_wo"],
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            weights["decode_post_rms_weight"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            inputs.logits,
        )

    def _pad_decode_inputs(self, model: RuntimeModel, inputs: _DecodeInputs) -> _DecodeKernelInputs:
        """Pad active decode rows to the fixed kernel batch.

        The fused decode kernel computes all ``max_batch_size`` rows. Inactive
        rows replicate row 0 so their KV writes are idempotent instead of
        targeting unrelated pages.
        """
        compiled = self._compiled
        actual_batch = inputs.actual_batch
        kernel_batch = model.runtime.max_batch_size
        max_blocks = self._max_blocks_per_seq(model)

        if kernel_batch > compiled.decode_logits_buffer.shape[0]:
            raise ValueError(
                f"kernel batch {kernel_batch} exceeds logits buffer batch "
                f"{compiled.decode_logits_buffer.shape[0]}"
            )

        hidden = compiled.decode_hidden_buffer
        hidden[:actual_batch].copy_(inputs.hidden)
        if actual_batch < kernel_batch:
            hidden[actual_batch:].copy_(inputs.hidden[0:1].expand(kernel_batch - actual_batch, -1))

        return _DecodeKernelInputs(
            actual_batch=actual_batch,
            hidden=hidden,
            seq_lens=self._copy_replicated_rows(
                compiled.decode_seq_lens_buffer,
                inputs.seq_lens,
                actual_batch,
                kernel_batch,
                rows_each=1,
            ),
            block_table=self._copy_replicated_rows(
                compiled.decode_block_table_buffer,
                inputs.block_table,
                actual_batch,
                kernel_batch,
                rows_each=max_blocks,
            ),
            slot_mapping=self._copy_replicated_rows(
                compiled.decode_slot_mapping_buffer,
                inputs.slot_mapping,
                actual_batch,
                kernel_batch,
                rows_each=1,
            ),
            logits=compiled.decode_logits_buffer,
        )

    def _run_distributed_program(self, callable_spec: _L3Callable, *args: Any) -> Any:
        """Run a compiled HOST wrapper through the shared PyPTO L3 worker."""
        span_args = {
            "kernel": callable_spec.name,
            "block_dim": callable_spec.block_dim,
            "aicpu_thread_num": callable_spec.aicpu_thread_num,
        }
        with profile_span(
            _kernel_trace_name(callable_spec.name),
            cat="kernel",
            level="kernel",
            args=span_args,
        ):
            worker = self._shared_l3_worker()
            l3_args = callable_spec.dispatch_args + tuple(self._coerce_l3_arg(worker, arg) for arg in args)
            worker_run_args = dict(span_args)
            with profile_span(
                f"{_kernel_trace_name(callable_spec.name)}.worker_run",
                cat="kernel",
                level="kernel",
                args=worker_run_args,
            ):
                timing = worker.run(callable_spec.compiled, *l3_args)
                _add_run_timing_args(worker_run_args, timing)
            _add_run_timing_args(span_args, timing)
            return timing

    def _shared_l3_worker(self) -> Any:
        """Return the worker shared by the generation prefill/decode path."""
        worker = self._l3_worker
        if worker is None:
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            worker = DistributedWorker([self._compiled.prefill.compiled, self._compiled.decode.compiled])
            self._l3_worker = worker
        return worker

    def _coerce_l3_arg(self, worker: Any, arg: Any) -> Any:
        """Convert static upload markers to worker-resident tensors."""
        if not isinstance(arg, _StaticDeviceTensor):
            return arg
        tensor = arg.tensor
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        cached = self._l3_static_tensors.get(key)
        if cached is not None:
            return cached
        dev = worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
        self._l3_static_tensors[key] = dev
        return dev

    def _materialize_static_tensors(self) -> None:
        """Upload static kernel tensors into the shared L3 worker before serving."""
        worker = self._shared_l3_worker()
        static = self._require_static_args()
        for arg in (
            static.final_norm_weight,
            static.rope_cos,
            static.rope_sin,
            static.padded_lm_head_weight,
            *static.decode_weights.values(),
        ):
            self._coerce_l3_arg(worker, arg)

    @staticmethod
    def _copy_replicated_rows(
        dst: torch.Tensor,
        active: torch.Tensor,
        actual_batch: int,
        kernel_batch: int,
        *,
        rows_each: int,
    ) -> torch.Tensor:
        """Copy active rows and fill inactive rows by replicating row 0."""
        active_view = active.reshape(actual_batch, rows_each)
        dst_view = dst.reshape(kernel_batch, rows_each)
        dst_view[:actual_batch].copy_(active_view)
        if actual_batch < kernel_batch:
            dst_view[actual_batch:].copy_(active_view[0:1].expand(kernel_batch - actual_batch, rows_each))
        return dst

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> _StaticDeviceTensor:
        """Mark a CPU tensor for one-time upload to the shared worker."""
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return _StaticDeviceTensor(Qwen314BModelRunner._share_cpu_tensor(tensor))

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    def close(self) -> None:
        """Release shared L3 worker resources and clear static tensor caches."""
        try:
            self.close_kv_cache()
        finally:
            worker = self._l3_worker
            try:
                if worker is not None:
                    worker.close()
            finally:
                self._l3_worker = None
                self._l3_static_tensors.clear()

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        compiled = self._compiled
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        max_seq = model.runtime.max_seq_len
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)
        kernel_batch = model.runtime.max_batch_size

        seq_len_values = [int(batch.seq_lens[idx].item()) for idx in range(actual_batch)]
        chunk_len_values: list[int] = []
        chunk_start_values: list[int] = []
        for batch_idx, seq_len in enumerate(seq_len_values):
            if batch.positions is not None:
                row_positions = batch.positions[batch_idx].detach().cpu()
                valid_positions = row_positions[row_positions >= 0]
                if valid_positions.numel() == 0:
                    raise ValueError("prefill positions must include at least one chunk token")
                chunk_start = int(valid_positions[0].item())
                chunk_len = int(valid_positions.numel())
                expected_positions = torch.arange(
                    chunk_start,
                    chunk_start + chunk_len,
                    dtype=valid_positions.dtype,
                )
                if not torch.equal(valid_positions, expected_positions):
                    raise ValueError(
                        "prefill batch.positions must form one contiguous chunk: "
                        f"chunk_start={chunk_start}, chunk_len={chunk_len}, seq_len={seq_len}"
                    )
            else:
                chunk_len = seq_len
                chunk_start = 0
            if chunk_len <= 0:
                raise ValueError("prefill chunk_lens must be positive")
            if chunk_start + chunk_len != seq_len:
                raise ValueError(
                    "prefill chunk must end at seq_len: "
                    f"chunk_start={chunk_start}, chunk_len={chunk_len}, seq_len={seq_len}"
                )
            chunk_len_values.append(chunk_len)
            chunk_start_values.append(chunk_start)
        total_tokens = sum(chunk_len_values)
        max_tokens = kernel_batch * max_seq
        if total_tokens > max_tokens:
            raise ValueError(f"prefill total tokens {total_tokens} exceeds kernel capacity {max_tokens}")

        hidden = compiled.prefill_hidden_buffer[:total_tokens]
        seq_lens = compiled.prefill_seq_lens_buffer
        chunk_lens = compiled.prefill_chunk_lens_buffer
        chunk_offsets = compiled.prefill_chunk_offsets_buffer
        block_table = compiled.prefill_block_table_buffer
        slot_mapping = compiled.prefill_slot_mapping_buffer[:total_tokens]
        seq_lens.zero_()
        chunk_lens.zero_()
        chunk_offsets.zero_()
        block_table.fill_(-1)

        token_offset = 0
        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = seq_len_values[batch_idx]
            if seq_len <= 0:
                raise ValueError("prefill seq_lens must be positive")
            if seq_len > max_seq:
                raise ValueError(f"prefill seq_len {seq_len} exceeds max_seq_len {max_seq}")
            seq_lens[batch_idx] = seq_len
            chunk_len = chunk_len_values[batch_idx]
            chunk_start = chunk_start_values[batch_idx]
            chunk_lens[batch_idx] = chunk_len
            chunk_offsets[batch_idx] = token_offset
            embeddings = batch.input_embeddings[batch_idx, :chunk_len, :].to(torch.bfloat16).cpu()
            hidden[token_offset : token_offset + chunk_len, :] = embeddings

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_block_table_row(block_table, batch_idx, max_blocks, page_ids)

            slot_row = self._compute_slot_mapping(page_ids, chunk_len, page_size, start_pos=chunk_start)
            slot_mapping[token_offset : token_offset + chunk_len] = slot_row
            token_offset += chunk_len

        return _PrefillInputs(
            actual_batch=actual_batch,
            hidden=hidden,
            seq_lens=seq_lens,
            chunk_lens=chunk_lens,
            chunk_offsets=chunk_offsets,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> _DecodeInputs:
        """Pack active decode requests into fused decode-kernel inputs."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        hidden_size = model.config.hidden_size
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)

        hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.empty((actual_batch,), dtype=torch.int32)

        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            if seq_len > model.runtime.max_seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} exceeds max_seq_len {model.runtime.max_seq_len}"
                )
            hidden[batch_idx, :] = batch.hidden_states[batch_idx].to(torch.bfloat16).cpu()
            seq_lens[batch_idx] = seq_len

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_block_table_row(block_table, batch_idx, max_blocks, page_ids)

            tokens_used = seq_len - 1
            page_idx = tokens_used // page_size
            offset = tokens_used % page_size
            slot_mapping[batch_idx] = page_ids[page_idx] * page_size + offset

        return _DecodeInputs(
            actual_batch=actual_batch,
            hidden=hidden,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    @staticmethod
    def _compute_slot_mapping(
        page_ids: list[int],
        num_tokens: int,
        page_size: int,
        *,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Return physical slot indices for token positions start_pos..start_pos+num_tokens-1."""
        mapping = torch.empty((num_tokens,), dtype=torch.int32)
        if num_tokens > 0:
            max_pos = start_pos + num_tokens - 1
            max_page_idx = max_pos // page_size
            if max_page_idx >= len(page_ids):
                raise ValueError(
                    f"page_ids list length {len(page_ids)} is too small for position {max_pos}; "
                    f"need at least {max_page_idx + 1} pages"
                )
        for offset_idx in range(num_tokens):
            pos = start_pos + offset_idx
            page_idx = pos // page_size
            offset = pos % page_size
            mapping[offset_idx] = page_ids[page_idx] * page_size + offset
        return mapping

    @staticmethod
    def _write_block_table_row(
        block_table: torch.Tensor,
        batch_idx: int,
        max_blocks: int,
        page_ids: list[int],
    ) -> None:
        """Write one request's KV page IDs into a flat block table."""
        row_start = batch_idx * max_blocks
        if page_ids:
            block_table[row_start : row_start + len(page_ids)] = torch.tensor(
                page_ids,
                dtype=torch.int32,
            )

    @staticmethod
    def _validate_batch_size(
        model: RuntimeModel,
        actual_batch: int,
    ) -> int:
        """Validate and return the actual user batch size."""
        if actual_batch <= 0:
            raise ValueError("batch must contain at least one request")
        if actual_batch > model.runtime.max_batch_size:
            max_batch_size = model.runtime.max_batch_size
            raise ValueError(
                f"batch has {actual_batch} requests, but runtime max_batch_size is {max_batch_size}"
            )
        return actual_batch

    @staticmethod
    def _max_blocks_per_seq(model: RuntimeModel) -> int:
        """Return the maximum KV pages one sequence can occupy."""
        return (model.runtime.max_seq_len + model.runtime.page_size - 1) // model.runtime.page_size
