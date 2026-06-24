"""
压力测试 + Profiling 脚本
- 发送 seq_len=4096 的长 prompt
- 并发请求
- 自动开关 profile
"""
import asyncio
import os
import time
import json
import urllib.request

BASE_URL = "http://127.0.0.1:8113"
MODEL = "qwen3"
OUTPUT_TOKENS = 128   # 输出 128 tokens（3500+128=3628 < max_model_len=5500）
CONCURRENCY = 16      # 对齐 decode 内核 BATCH=16，全程保持满员
TOTAL_REQUESTS = 32   # 跑 2 轮，结果更稳定

# trace 解析相关：profiler 产物在容器内（root 权限），经 docker exec 读取
CONTAINER = "vllm-qwen3"
PROFILE_DIR = "/data/l00955553/vllm_profile"

# 推理生成文本落盘到 txt。None=自动按时间戳命名，放 PROFILE_DIR（vllm_profile）下；
# 也可写成固定路径，如 "/data/l00955553/infer_output.txt"。
OUTPUT_TXT = None

# 输出详细程度总开关：
#   False（默认）= 简略，只打印 decode 的几个关键指标（TPOT/device占用/吞吐/算子三桶）。
#   True          = 详细，额外打印逐请求行、prefill 明细、步周期分布表、逐 op 全表、
#                   批次分布、步周期直方图、预热 vs 稳态。
VERBOSE = False

# 长文本 prompt 从外部文件读取（不再写死在脚本里）。
# 路径优先级：命令行 -p/--prompt  >  环境变量 PROMPT_FILE  >  默认路径
# 默认：profile-util/prompt/long_prompt.txt（相对脚本定位，换机器也不用改）。
_DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "prompt", "long_prompt_3.5k.txt")
PROMPT_FILE = os.environ.get("PROMPT_FILE", _DEFAULT_PROMPT_FILE)


def _load_prompt(path: str | None = None) -> str:
    """读取外部 prompt 文件（path=None 时用当前 PROMPT_FILE）；找不到给清晰报错。"""
    path = path or PROMPT_FILE
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise SystemExit(f"[prompt] 找不到 prompt 文件: {os.path.abspath(path)}\n"
                         f"请创建它，或用 -p/--prompt、环境变量 PROMPT_FILE 指定。")


LONG_PROMPT = _load_prompt()



# 绕过 http_proxy/https_proxy/all_proxy：BASE_URL 是本机 vllm，
# 经代理会被转发并返回 502 Bad Gateway。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, data: dict | None = None, timeout: int = 180) -> dict:
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw.strip() else {}


def count_tokens(text: str) -> int | None:
    """用 vLLM /tokenize 接口取文本的真实 token 数；失败返回 None。"""
    try:
        return post(f"{BASE_URL}/tokenize", {"model": MODEL, "prompt": text}).get("count")
    except Exception:
        return None


def start_profile():
    try:
        post(f"{BASE_URL}/start_profile")
        print("[profile] Started")
    except Exception as e:
        print(f"[profile] Start failed: {e}")


def stop_profile():
    try:
        post(f"{BASE_URL}/stop_profile")
        print("[profile] Stopped, files saved to /data/l00955553/vllm_profile/")
    except Exception as e:
        print(f"[profile] Stop failed: {e}")


async def send_request(session_id: int, sem: asyncio.Semaphore) -> dict:
    async with sem:
        loop = asyncio.get_event_loop()
        payload = {
            "model": MODEL,
            "prompt": LONG_PROMPT,
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            result = await loop.run_in_executor(
                None, lambda: post(f"{BASE_URL}/v1/completions", payload)
            )
            elapsed = time.perf_counter() - t0
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            if VERBOSE:
                print(f"  [req {session_id:02d}] prompt={prompt_tokens}t "
                      f"output={completion_tokens}t time={elapsed:.2f}s")
            return {"id": session_id, "elapsed": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "text": result.get("choices", [{}])[0].get("text", ""),
                    "ok": True}
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [req {session_id:02d}] ERROR: {e} ({elapsed:.2f}s)")
            return {"id": session_id, "elapsed": elapsed, "ok": False}


async def main() -> float:
    """返回测试开始时间戳（UTC）"""
    _ntok = count_tokens(LONG_PROMPT)
    _in = f"~{_ntok}" if _ntok else f"{len(LONG_PROMPT)}字符"
    print("=== vLLM 压力测试 + Profiling ===")
    print(f"  模型: {MODEL}  并发: {CONCURRENCY}  总请求: {TOTAL_REQUESTS}")
    print(f"  输入: {_in} tokens  输出: {OUTPUT_TOKENS} tokens")
    print()

    start_profile()
    test_start_ts = time.time()   # 记录本次测试开始的 UTC 时间戳

    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.perf_counter()
    results = await asyncio.gather(*[send_request(i + 1, sem) for i in range(TOTAL_REQUESTS)])
    total_elapsed = time.perf_counter() - t_start

    stop_profile()

    ok = [r for r in results if r["ok"]]
    if ok:
        lats = sorted(r["elapsed"] for r in ok)
        total_out = sum(r["completion_tokens"] for r in ok)
        print()
        print("=== 压测结果 ===")
        print(f"  成功: {len(ok)}/{TOTAL_REQUESTS}  总耗时: {total_elapsed:.2f}s")
        print(f"  延迟  avg={sum(lats)/len(lats):.2f}s  "
              f"p50={lats[len(lats)//2]:.2f}s  p90={lats[int(len(lats)*0.9)]:.2f}s")
        print(f"  吞吐: {total_out/total_elapsed:.1f} output tokens/s")

    # 把每个请求的【推理生成文本】落盘到 txt
    _save_outputs(results)

    return test_start_ts, total_elapsed


def _save_outputs(results: list):
    """把各请求的推理生成文本写入 txt（默认放 PROFILE_DIR 下，按时间戳命名）。"""
    f, path = _open_output()
    with f:
        f.write(f"=== 推理输出  模型={MODEL}  并发={CONCURRENCY}  "
                f"请求={TOTAL_REQUESTS}  每请求≤{OUTPUT_TOKENS} tokens ===\n\n")
        for r in sorted(results, key=lambda x: x["id"]):
            if not r.get("ok"):
                f.write(f"[req {r['id']:02d}] ERROR\n\n")
                continue
            f.write(f"[req {r['id']:02d}] prompt={r.get('prompt_tokens', 0)}t "
                    f"output={r.get('completion_tokens', 0)}t\n")
            f.write((r.get("text") or "").strip() + "\n")
            f.write("-" * 60 + "\n\n")
    print(f"[output] 推理输出已保存到: {path}")


def _pct(sorted_vals: list, q: float) -> float:
    """线性插值分位数，q ∈ [0, 1]，输入需已排序"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stats(vals: list) -> dict:
    """返回一组数值的常用统计量"""
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    std = var ** 0.5
    return {
        "n": n, "min": s[0], "max": s[-1], "mean": mean, "std": std,
        "cv": (std / mean * 100) if mean else 0.0,
        "p50": _pct(s, 0.50), "p90": _pct(s, 0.90),
        "p95": _pct(s, 0.95), "p99": _pct(s, 0.99),
    }


def _histogram(vals: list, bins: int = 10, width: int = 40) -> list:
    """生成 ASCII 直方图行，返回字符串列表"""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [f"  {lo:8.2f} ms | {'#' * width} ({len(vals)})"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        edge_lo = lo + i * step
        edge_hi = edge_lo + step
        bar = "#" * int(c / peak * width)
        lines.append(f"  [{edge_lo:7.2f}, {edge_hi:7.2f}) ms | {bar:<{width}} {c}")
    return lines


def analyze_logs(test_start_ts: float):
    """从 docker logs 中提取 prefill/decode 耗时，只统计本次测试的数据"""
    import subprocess, re
    from datetime import datetime, timezone

    result = subprocess.run(
        ["docker", "logs", "vllm-qwen3", "--timestamps"],
        capture_output=True, text=True
    )
    logs = result.stdout + result.stderr

    # docker --timestamps 格式: 2026-06-17T09:22:13.123456789Z  <log line>
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    iter_pattern = re.compile(
        r"Iteration\(\d+\): (\d+) context requests, (\d+) context tokens, "
        r"(\d+) generation requests, (\d+) generation tokens, "
        r"iteration elapsed time: ([0-9.]+) ms"
    )

    prefill_times, decode_times = [], []
    skipped = 0

    for line in logs.splitlines():
        # 解析行首时间戳
        ts_m = ts_pattern.match(line)
        if ts_m:
            try:
                line_dt = datetime.strptime(ts_m.group(1), "%Y-%m-%dT%H:%M:%S")
                line_ts = line_dt.replace(tzinfo=timezone.utc).timestamp()
                if line_ts < test_start_ts:
                    skipped += 1
                    continue
            except ValueError:
                pass

        m = iter_pattern.search(line)
        if not m:
            continue
        ctx_req, ctx_tok, gen_req, gen_tok, ms = (
            int(m.group(1)), int(m.group(2)),
            int(m.group(3)), int(m.group(4)),
            float(m.group(5))
        )
        if ms < 1:
            continue
        if ctx_tok > 10:
            prefill_times.append((ctx_tok, ms))
        elif gen_req > 0 and ctx_req == 0:
            decode_times.append((gen_req, ms))

    print()
    print(f"=== Prefill / Decode 详细分析（本次测试，已过滤 {skipped} 条历史行）===")
    if prefill_times:
        avg_tps = sum(t / m * 1000 for t, m in prefill_times) / len(prefill_times)
        print(f"\n[Prefill] 本次共 {len(prefill_times)} 次，平均 prefill 速度: {avg_tps:.0f} tokens/s")
        if VERBOSE:
            for tok, ms in prefill_times[-10:]:
                tps = tok / (ms / 1000)
                print(f"  {tok:6d} tokens  {ms:8.2f} ms  ({tps:.0f} prefill tokens/s)")

    # 注意：decode 单步耗时不在这里统计。
    # iteration elapsed time 日志会系统性低估单步（计时器在 execute_model 异步下发
    # 之后才启动，只量到 device 尾段；async ON 时低估更严重）。
    # decode 的真实耗时一律由 analyze_trace() 从 Ascend Profiler trace 计算。
    if decode_times:
        print(f"\n[Decode] 本次 {len(decode_times)} 个 decode 步——真实耗时见末尾"
              f"【Ascend Profiler trace】分析（此处不再用 iteration log 口径）。")


def _dexec(cmd: str, timeout: int = 120):
    """在容器内执行一条命令（profiler 产物是 root 权限，宿主机直接读不了）。"""
    import subprocess
    return subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _newest_trace_dir() -> str | None:
    """取最新的 rank0_*_ascend_pt 目录（每次 start/stop_profile 生成一个）。"""
    r = _dexec(f"ls -dt {PROFILE_DIR}/rank0_*_ascend_pt 2>/dev/null | head -1")
    return r.stdout.strip() or None


def _fix_perms(path: str = PROFILE_DIR):
    """profiler 产物在容器内是 root:root 700，宿主机非 root 用户读不到。
    用 chmod -R a+rX 放开：目录加可进入(x)、文件加可读(r)，仅改元数据，很快。"""
    try:
        _dexec(f"chmod -R a+rX {path} 2>/dev/null; true", timeout=120)
        if VERBOSE:
            print(f"[profile] 已放开读权限（容器外可访问）: {path}")
    except Exception as e:
        print(f"[profile] 调整权限失败: {e}")


def _interval_union_us(intervals) -> float:
    """合并区间求并集长度（去掉 kernel 之间的重叠/空隙后真正占用的时间）。"""
    m = 0.0
    cs = ce = None
    for x, y in sorted(intervals):
        if cs is None:
            cs, ce = x, y
        elif x <= ce:
            ce = max(ce, y)
        else:
            m += ce - cs
            cs, ce = x, y
    if cs is not None:
        m += ce - cs
    return m


def analyze_trace(test_start_ts: float, measured_wall: float):
    """从 Ascend Profiler 的 kernel_details.csv 计算【真实】decode 单步耗时与分布。

    为什么不用 iteration elapsed time 日志：那个计时器在 `execute_model(non_block=True)`
    把 kernel 异步下发之后才启动，只量到 device 计算的尾段，系统性低估单步（async ON
    时更严重）。trace 里每个 kernel 的 device 起止是硬件计数器，且 trace 总跨度与压测
    墙钟吻合（本函数打印该校验），所以是权威口径。

    切步：用 ArgMaxV2（每个调度步采样一次）切分。
    判定 decode 步：窗口内有 MatMulV2 且无 MatMulV3（prefill 的大 GEMM）——与 async
    开关无关，不依赖步周期阈值。
    单步真实耗时 = 相邻 ArgMaxV2 间隔（步周期，= TPOT）；device 占用 = 步内 kernel 区间并集。
    batch 大小取自日志的 generation requests 计数（计数准确，只有时间被低估）。
    """
    import subprocess, csv as _csv, io, re as _re, time as _time, statistics as S, bisect
    from collections import defaultdict
    from datetime import datetime, timezone

    print("\n" + "=" * 64)
    print("=== Decode 真实耗时（来自 Ascend Profiler trace）===")

    tdir = _newest_trace_dir()
    if not tdir:
        print("  未找到 trace 目录；确认 start_profile/stop_profile 已执行。")
        return
    out = f"{tdir}/ASCEND_PROFILER_OUTPUT"
    kernel_csv = f"{out}/kernel_details.csv"

    # 打印本次实际解析的 trace 路径与文件（仅详细模式）
    if VERBOSE:
        print(f"  trace 目录:   {tdir}")
        print(f"  解析文件:     {kernel_csv}")
        print(f"  (耗时来源)    docker logs {CONTAINER}  —— 仅取 batch 计数")

    # 等待 CANN 后处理生成 analyse.done（stop_profile 后通常还要 ~30-60s）
    for _ in range(60):
        if _dexec(f"test -f {out}/analyse.done && echo ok").stdout.strip() == "ok":
            break
        print("  等待 profiler 解析（analyse.done）…")
        _time.sleep(5)

    # 产物已齐全，放开权限供容器外访问
    _fix_perms(tdir)

    raw = subprocess.run(
        ["docker", "exec", CONTAINER, "cat", kernel_csv],
        capture_output=True, text=True, timeout=300,
    ).stdout
    rows = []
    for d in _csv.DictReader(io.StringIO(raw)):
        try:
            st = float(d['Start Time(us)'].strip().rstrip('\t').strip())
            du = float(d['Duration(us)'])
        except (ValueError, KeyError):
            continue
        rows.append((st, du, d['Type']))
    if not rows:
        print("  kernel_details.csv 为空。")
        return
    rows.sort()
    starts = [r[0] for r in rows]

    # 校验：trace 跨度应 ≈ 压测墙钟（证明 trace 没漏、没被拉伸）
    span = (max(st + du for st, du, _ in rows) - rows[0][0]) / 1e6
    print(f"  校验: trace 跨度 {span:.2f}s  ←→  压测墙钟 {measured_wall:.2f}s  "
          f"(差 {abs(span - measured_wall):.2f}s)")

    marks = [st for st, du, t in rows if t == 'ArgMaxV2']
    if len(marks) < 2:
        print("  trace 中未找到足够的 ArgMaxV2 步标记。")
        return
    bounds = [(marks[i - 1], marks[i]) for i in range(1, len(marks))]

    # 逐步分类并统计：decode 步 = 有 MatMulV2 且无 MatMulV3
    periods, devs, n_other = [], [], 0
    op_l = defaultdict(list)        # 算子桶 -> 每步 device 时长(ms)
    op_time_sum = defaultdict(float)  # 各 op 全程 device 总时长(us)，用于逐 op 全表
    op_cnt_sum = defaultdict(int)     # 各 op 全程总次数
    for a, b in bounds:
        sub = rows[bisect.bisect_left(starts, a):bisect.bisect_left(starts, b)]
        types = {t for _, _, t in sub}
        if 'MatMulV3' in types or 'MatMulV2' not in types:
            n_other += 1
            continue
        periods.append((b - a) / 1000.0)
        devs.append(_interval_union_us((st, st + du) for st, du, _ in sub) / 1000.0)
        bucket = defaultdict(float)
        for st, du, t in sub:
            op_time_sum[t] += du
            op_cnt_sum[t] += 1
            if t == 'MatMulV2':
                bucket['线性层 MatMul'] += du
            elif t == 'FusedInferAttentionScore':
                bucket['注意力 Attention'] += du
            else:
                bucket['其它(norm/rope/激活/采样)'] += du
        for k, v in bucket.items():
            op_l[k].append(v / 1000.0)
    if not periods:
        print("  未识别到 decode 步（每个窗口都含 prefill GEMM）。")
        return

    # batch 大小：从 iteration 日志取 generation requests 计数（仅计数准确，时间不取用）
    logs = subprocess.run(["docker", "logs", CONTAINER, "--timestamps"],
                          capture_output=True, text=True).stdout
    pat = _re.compile(r"Iteration\(\d+\): (\d+) context requests, \d+ context tokens, "
                      r"(\d+) generation requests, \d+ generation tokens, "
                      r"iteration elapsed time: ([0-9.]+) ms")
    tsp = _re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    batches = []
    for line in logs.splitlines():
        tm = tsp.match(line)
        if tm:
            try:
                lt = datetime.strptime(tm.group(1), "%Y-%m-%dT%H:%M:%S").replace(
                    tzinfo=timezone.utc).timestamp()
                if lt < test_start_ts:
                    continue
            except ValueError:
                pass
        m = pat.search(line)
        if m and int(m.group(1)) == 0 and int(m.group(2)) > 0 and float(m.group(3)) >= 1:
            batches.append(int(m.group(2)))

    # ── 汇总（步周期 = 真实单步；全程 vs 采样末 20 步）──
    second = periods[CONCURRENCY:] if len(periods) > CONCURRENCY else periods
    sample = second[-20:] if len(second) >= 20 else second
    st_all, st_s = _stats(periods), _stats(sample)
    mean_p = st_all['mean']
    dev_mean = S.mean(devs)
    occ = dev_mean / mean_p * 100

    batch_str = f"{S.mean(batches):.1f}/{CONCURRENCY}" if batches else "n/a"

    # ───────── 关键指标（简略 & 详细都打印）─────────
    print(f"  decode 步数 {len(periods)}   平均 batch {batch_str}")
    print("  ── 关键指标 ──")
    print(f"  TPOT(真实单步):  {mean_p:6.2f} ms  (p50 {st_all['p50']:.2f}, p99 {st_all['p99']:.2f})")
    print(f"  device 计算:     {dev_mean:6.2f} ms  (NPU 占用 {occ:.1f}%)")
    print(f"  host 间隙:       {mean_p - dev_mean:6.2f} ms")
    print(f"  整 batch 吞吐:   {CONCURRENCY * 1000 / mean_p:6.1f} tok/s   单请求 {1000 / mean_p:.1f} tok/s")

    if not VERBOSE:
        return

    # ───────── 以下为详细输出（VERBOSE=True）─────────
    print("\n  ── 算子分解 ──")
    tot = sum(S.mean(v) for v in op_l.values())
    for k in sorted(op_l, key=lambda x: -S.mean(op_l[x])):
        ms = S.mean(op_l[k])
        print(f"  {k:<28} {ms:6.2f} ms ({ms / tot * 100:4.1f}%)")

    print("\n  ── 单步真实耗时 = 步周期（全程 vs 采样末20步）──")
    print(f"  {'指标':<10}{'全程':>14}{'采样':>14}")
    print(f"  {'步数':<12}{st_all['n']:>12}{st_s['n']:>14}")
    for key in ('min', 'mean', 'p50', 'p90', 'p95', 'p99', 'max', 'std'):
        print(f"  {key:<12}{st_all[key]:>11.2f}ms{st_s[key]:>11.2f}ms")
    print(f"  {'抖动 CV':<10}{st_all['cv']:>12.1f}%{st_s['cv']:>13.1f}%")

    n_dec = len(periods)
    tot_us = sum(op_time_sum.values())

    def _bk(t):
        if t == 'MatMulV2':
            return '线性层 MatMul'
        if t == 'FusedInferAttentionScore':
            return '注意力 Attention'
        return '其它'

    print("\n  ── 逐 op 全表（decode 步平均：次数/步、device 时长/步）──")
    print(f"  核对：MatMulV2≈层数×4+1(lm_head)，多数 op 次数≈层数，可验证桶分正确")
    print(f"  {'op type':<32}{'次数/步':>9}{'ms/步':>10}{'占比%':>8}   桶")
    for t, us in sorted(op_time_sum.items(), key=lambda x: -x[1]):
        print(f"  {t:<32}{op_cnt_sum[t] / n_dec:>9.1f}"
              f"{us / n_dec / 1000:>10.3f}{us / tot_us * 100:>8.1f}   {_bk(t)}")
    print(f"  {'合计(=device 累加)':<32}{sum(op_cnt_sum.values()) / n_dec:>9.1f}"
          f"{tot_us / n_dec / 1000:>10.3f}{100.0:>8.1f}")

    if batches:
        print("\n  ── 批次大小分布（日志计数）──")
        bh = {}
        for bb in batches:
            bh[bb] = bh.get(bb, 0) + 1
        for bb in sorted(bh):
            bar = "#" * int(bh[bb] / max(bh.values()) * 40)
            print(f"  batch={bb:2d}: {bar:<40} {bh[bb]:4d} 步 ({bh[bb] / len(batches) * 100:4.1f}%)")

    print("\n  ── 步周期直方图（全程，真实）──")
    for hl in _histogram(periods, bins=10, width=40):
        print(hl)

    if len(periods) > CONCURRENCY:
        warm, steady = periods[:CONCURRENCY], periods[CONCURRENCY:]
        w, s = S.mean(warm), S.mean(steady)
        print("\n  ── 预热 vs 稳态 ──")
        print(f"  预热段(前{CONCURRENCY}步):  avg {w:.2f} ms/step")
        print(f"  稳态段(后{len(steady)}步): avg {s:.2f} ms/step")
        print(f"  稳态相对预热:        {(s / w - 1) * 100:+.1f}%")


def _open_output():
    """返回 (文件对象, 路径)；None 时按时间戳自动命名到 PROFILE_DIR（vllm_profile）下。"""
    import os, datetime
    path = OUTPUT_TXT
    if not path:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        path = os.path.join(PROFILE_DIR, f"infer_output_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "w", encoding="utf-8"), path


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="vLLM 压测 + Profiling（decode 真实耗时走 trace）")
    _ap.add_argument("-p", "--prompt", metavar="FILE",
                     help="prompt 文件路径（覆盖环境变量 PROMPT_FILE 与默认）")
    _args = _ap.parse_args()
    if _args.prompt:                  # 命令行优先级最高
        PROMPT_FILE = _args.prompt
        LONG_PROMPT = _load_prompt(PROMPT_FILE)
    print(f"[prompt] 使用: {os.path.abspath(PROMPT_FILE)}")

    test_start_ts, measured_wall = asyncio.run(main())
    if VERBOSE:                       # prefill/decode 日志明细仅详细模式打印
        analyze_logs(test_start_ts)
    analyze_trace(test_start_ts, measured_wall)
