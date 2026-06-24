# 性能优化记录：decode logits 只拷回活跃行

日期：2026-06-08
模型 / 场景：Qwen3-14B，serving 单请求 decode（a2a3 tensormap_and_ringbuffer runtime）
改动文件：`examples/model/qwen3_14b/runner/npu_runner.py`（decode forward）

## 背景与定位

通过给 runtime 的 `[chip_timing]` 树逐层打点（`validate` 拆出 `status` / `copy_back` /
`free_cleanup`，`prebuilt_arena` 拆出 `arena_build` / `arena_upload`，并给 `copy_back`
加上字节数与有效带宽），定位到稳态 decode 每步的主机侧开销集中在两处对称的浪费：

- `validate.copy_back`：**1.81 ms**，D2H 拷回 **3 个 tensor pair、9.438 MB**，~5 GB/s。
- `bind.args_malloc_copy`：**1.90 ms**，对应这些 tensor 每步的 H2D 上传。

进一步分析确认这 9.438 MB ≈ decode logits buffer `[batch=16, vocab=152064] × fp32`
= 9.28 MB（其余 0.16 MB 是 hidden，seq_lens 极小）。

根因：decode kernel 的 batch 被固定编译为 16（`pypto-lib/models/qwen3/14b/config.py:23`
`BATCH=16`），但单请求时只有 1 行有效；logits buffer 作为**普通输出 tensor** 传给
kernel，runtime 因此每步都：

1. 把整块 9.28 MB H2D 上传（`args_malloc_copy` 里的浪费，输出 buffer 根本不需要上传）；
2. 把整块 9.28 MB D2H 拷回（`copy_back`），而 host 随后只 `index_select` 出活跃那 1 行
   （`npu_runner.py` 中 `logits_padded.index_select(0, slot_rows)`）。

即 **~94% 的搬运字节是白费的**，且上传方向同样冤枉。

## 改动内容

把 logits buffer 改为**设备常驻的 child_memory 输出**，并在 run 之后只 D2H 拷回活跃
slot 覆盖的连续前缀：

- `logits_dev = self._l2_child_tensor(rt, logits_padded, upload=False)`，以 child_memory
  形式传给 kernel —— runtime 不再对它自动 H2D/D2H。
- kernel 照常把全部 `kernel_batch` 行写入这块常驻 device buffer。
- run 完后用 `worker.copy_from(host, dev, rows_needed * row_nbytes)` 只拷回
  `[0 : max(slot)+1]` 行。slot 低位优先分配且跨步稳定（`_assign_decode_slots`），所以所有
  活跃行必落在这个前缀内；之后照旧 `index_select` 取各请求的行。

不动 kernel、不改 runtime C++、不改 prefill 路径。纯 serving 侧 Python 改动。

## 实测效果

环境：单请求，prompt 固定，batch padding=16，生成 5 token。稳态 decode 取 run2–4
（排除 run1 的 `so_load` dlopen 预热）。两次对照为同一脚本前后各跑一次（N=1）。

| 指标（稳态 decode / 步） | 改动前 | 改动后 | 变化 |
|---|---:|---:|---:|
| `validate.copy_back` | 1.81 ms（9.438 MB，3 pairs） | 0.04 ms（0.156 MB，2 pairs） | −98% |
| `validate`（合计） | ~1.85 ms | 0.06 ms | −97% |
| `bind.args_malloc_copy` | ~1.90 ms | 0.09 ms | −95% |
| `bind_impl`（合计） | ~3.53 ms | ~1.69 ms | −52% |
| chip step total | ~42.4 ms | ~38.3 ms | −9.7%（约 −4 ms/步） |
| decode 吞吐 | 3.65 tok/s（273.9 ms/tok） | 3.75 tok/s（266.6 ms/tok） | +2.7% |

合计每步省下约 3.6 ms 主机侧搬运（~1.8 ms D2H + ~1.8 ms H2D），构成了 chip step
~4 ms 的下降。新增的显式前缀读回（单请求 ~0.58 MB，~0.1 ms）发生在 Python 侧、不计入
chip_timing 树。

正确性：token 输出与改动前完全一致
（`[264, 8453, 67926, 5440, 2813]` / "a Chinese multinational technology company"）。

## 说明与注意

- **chip-step 降幅（−9.7%）大于端到端 decode 降幅（−2.7%）**：`ms/token` 还含 chip 步之外
  的 Python 开销（采样、embedding、slot 管理等），那部分本次未动。
- **e2e / TTFT 数字本次偏噪**：本轮 prefill `weight_upload` 6600 ms vs 上一轮 5041 ms
  （init/prefill 的 run-to-run 方差），导致 overall tok/s 0.60→0.50。这**不是本改动的回退**
  ——本改动只影响 decode 路径，decode 指标全部改善。
- **退化边界（仍正确）**：若请求频繁进出导致活跃 slot 稀疏且最大 slot 很高（如只剩
  slot 15），该步 `rows_needed=16` 退化为全拷，无收益但结果正确。单 / 少请求的常见场景
  拿到全部收益。

## 后续可做（本次未做）

- **prefill 的 copy_back 仍是 1.28 GB / ~243 ms**（`dev=4 run=0`）：prefill 把所有位置的
  logits 都拷回，而采样只需最后一个位置一行，浪费量远大于 decode，属于一次性 TTFT 开销。
- `hidden` / `seq_lens` 作为输入仍每步 H2D + 被 D2H 拷回（~0.16 MB，很小）：可改 child_memory
  + `refresh=True`（只上传不拷回）进一步去掉那 0.16 MB 的拷回。
- decode 单步真正的大头仍是 device 侧 `runner_run.sync` ~36 ms（`graph_build` ~16 ms +
  `post_orch` ~19 ms，占一步 ~94%）。要显著提 decode 吞吐需攻 device 侧，本优化属于
  「便宜、干净、顺手」的主机侧收益。

## 复现 / 度量方式

打点输出见 runtime 的 `[chip_timing]` 树（`SIMPLER_CHIP_TIMING=1`）：
- `validate` 三段：`src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`
- `copy_back` 的字节/带宽与 `prebuilt_arena` 子项：同上 + `src/common/platform/include/common/chip_run_timing.h`
- 渲染：`src/common/platform/onboard/host/c_api_shared.cpp` 的 `print_chip_timing_tree`
（以上打点在 pypto-github/runtime 仓）。
