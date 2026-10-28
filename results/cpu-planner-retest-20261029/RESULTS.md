# CPU planner 补测（2026-10-29）

本补测只验证 DSpark 的 CPU planner 与真实 compact verify 路径，不把结果扩展成 cache、FP8、CUDA Graph 或语义质量结论。远端机器为 NVIDIA RTX PRO 6000 Blackwell Server Edition；模型为 Qwen3-14B，draft 为 `dspark_qwen3_14b_block7`。

## 固定条件

- BF16 target / BF16 KV，`--bf16-gemm-backend torch`；FP8、prefix cache、radix cache、CUDA Graph、compact CUDA scan 均关闭。
- `SGLANG_RAGGED_VERIFY_MODE=compact`，`SGLANG_SIMULATE_ACC_LEN=-1`，使用真实接受长度；记录 `core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,verify_path,reqs`。
- 每个 cell 先发 1 个不计入统计的 warmup，再发 2 波固定长度随机 token 请求；输出长度为 128。随机 token 只用于硬件成本和 planner 形状测量，不用于语义质量结论。
- 记录字段包括 budget、实际 verify_len、compact/full token 数、accept length、step CPU/GPU、draft GPU、target verify GPU、TTFT/TPOT/E2E。`step_cpu_ms` 是整个 decode step 的 CPU 间隔，不是只包住 `compute_verify_token_budget` 的独占时间。

## 真实 compact serving 矩阵

| 输入 token | 并发 | planner budget 均值 / 中位数 / p95 | graph token 均值 / p95 | accept length 均值 | step CPU ms 均值 | E2E 均值 / p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 7.00 / 7 / 7 | 8.00 / 8 | 6.27 | 65.04 | 1486.0 / 1422.8 |
| 512 | 4 | 23.71 / 28 / 28 | 27.08 / 32 | 3.77 | 60.36 | 3225.1 / 3366.8 |
| 512 | 8 | 29.89 / 21 / 56 | 34.16 / 64 | 4.63 | 58.89 | 2051.5 / 3638.5 |
| 512 | 16 | 43.19 / 28 / 112 | 49.34 / 128 | 3.45 | 57.05 | 3558.4 / 5080.4 |
| 2048 | 1 | 7.00 / 7 / 7 | 8.00 / 8 | 2.42 | 62.33 | 3534.7 / 3191.2 |
| 2048 | 4 | 22.89 / 21 / 28 | 26.09 / 32 | 2.67 | 58.85 | 3698.6 / 4208.9 |
| 2048 | 8 | 20.81 / 7 / 56 | 23.78 / 64 | 2.81 | 60.29 | 4375.9 / 9333.4 |
| 2048 | 16 | 51.42 / 42 / 112 | 58.76 / 128 | 3.22 | 58.07 | 5606.7 / 6594.2 |
| 8192 | 1 | 7.00 / 7 / 7 | 8.00 / 8 | 3.56 | 68.07 | 3506.6 / 2443.0 |
| 8192 | 4 | 20.66 / 21 / 28 | 23.61 / 32 | 3.52 | 64.69 | 6025.2 / 6996.5 |
| 8192 | 8 | 23.49 / 14 / 56 | 26.82 / 64 | 3.17 | 64.81 | 9869.7 / 12040.4 |

`p8192/c16` 未运行：服务的 `max_total_num_tokens=131072`，仅放 16×8192 prompt 就已耗尽 token capacity，不能把它写成性能失败。

## planner 微路径（CPU only）

对同一份 additive SPS table 和同一份 survival tensor，比较每次是否重新构造 additive step-time vector 与复用按 request 数缓存的 vector；两条路径的 budget 逐次相等。

| request 数 | candidate 数 | uncached median us | cached median us | median speedup |
|---:|---:|---:|---:|---:|
| 1 | 9 | 84.322 | 38.666 | 2.18× |
| 4 | 33 | 87.162 | 39.366 | 2.21× |
| 8 | 65 | 88.272 | 40.169 | 2.20× |
| 16 | 129 | 94.242 | 42.583 | 2.21× |

结果范围：uncached 中位数约 84.3–94.2 us，cached 中位数约 38.7–42.6 us，约 2.18–2.21×；这与旧报告的 90.322→39.205 us 方向一致，但本次是相同源码、同一 additive table 下的独立 CPU microbench。

## 结论边界

1. c1 在这组随机输入下始终选择 budget=7、verify=8；并发增大后预算出现阶梯变化，说明动态 planner 在真实 confidence / batch 形状下被触发。
2. 本补测没有比较普通 AR、full verify 与 dynamic verify 的语义任务质量，也没有证明 E2E 收益；E2E/TTFT/TPOT 只是同一服务条件下的观测量。
3. `predicted_step_ms` 来自已有 SPS table，而这次服务关闭 CUDA Graph；它不能直接当作本次 target GPU 实测时间。
4. 原始逐 cell JSON、planner microbench、启动脚本和服务日志均在本目录；复跑时应先核对 SGLang commit、模型、driver、SPS table SHA256 和 token capacity。

## 文件

- `matrix_summary.json`：汇总表。
- `matrix/`：逐 cell 的 responses、planner records 和观测字段。
- `planner_microbench.json`：CPU planner 缓存前后 microbench。
- `run_matrix.py`、`planner_microbench.py`：复现实验脚本。
- `launch.sh`、`server-info-final.json`、`gpu-final.csv`、`server.log`：运行条件与原始服务证据。
