# DSpark 60 秒停顿诊断：根因未确认，停止试改

日期：2026-10-08。诊断对象为 h61 baseline，不是 graph 候选。仅一次有上限的 h62 观测运行；发生崩溃后停止，无重试、无性能 A/B、无框架或 kernel 修复。

## 当前结论

原来的 60 秒停顿确实存在，但根因仍未确认。不能宣称已定位为编译、GC、网络盘或驱动问题，也不能把它归因于候选 graph。当前证据不足以支持大改 SGLang 框架。

新增加的异步线程栈观察器在 h62 中发生了 scheduler SIGSEGV，最后栈转储残缺。这条观测路径已经弃用；未将它合入服务配置。没有为了追查而连续更换协议、升级依赖或重启试验。

## h61 的原始停顿证据

- 22:28:22–22:29:22（UTC+8）服务日志空白约 60 秒，之前仍有 5 个运行请求和 5 个队列请求。
- GPU 连续 299 个样本中 294 个 utilization 为 0%；采样最大间隔 0.251 秒。不是整台机器采样中断。
- 15 个 c16 客户端请求延迟约 66–68 秒；有些首 token 已正常到达，停顿发生在 decode，有些 TTFT 达到 63–65 秒。不能用正常排队单独解释。
- 服务 max-running-requests=8；c16 是 16 个客户端与 8 个运行槽位，不是 GPU batch16。
- graph OFF、cache OFF、greedy、原固定 FA4 64×64 数值 baseline；所有历史已保存 token 和 5 项 acceptance 一致。
- 没有当时的 CPU/native 栈、逐秒 CPU 限流数据或可读取的内核日志。

## 只读核查与候选原因

| 方向 | 证据与限制 | 判断 |
|---|---|---|
| 网络存储/同步日志 IO | 实测 `/root/autodl-fs` 是 `fuse.autofs`，h61 服务日志、请求 JSONL 和测试工具在该盘；scheduler 同步输出日志，客户端同步写/读记录。AutoDL 明确说明它是网络共享存储，本地数据盘更适合高 IO。h62 中也采到两次 `request_wait_answer` 等待，但未复现原长停顿。 | 合理假设，未证明 |
| 冷 JIT | 时间窗内 `/root/.cache` 和 `/tmp/root/cutlass_python_cache` 无对应文件写入。FA4 默认是内存 JIT cache，磁盘无新文件不能排除编译。已有的 6144-token prefill 在停顿前出现过，不能只凭最后一条日志断定新 shape。 | 未排除，未证明 |
| GC | 原框架已在服务启动 warmup 后调用 freeze_gc；可能存在之后的对象分配，不能据此完全排除。h62 可见 GC 最长 0.440 秒，不是同一事件。 | 没有 60 秒证据 |
| CPU 限流/宿主调度 | cgroup CPU quota 是 25 CPU；累计有 throttled_usec。历史累计值没有时间归属，不能把它直接当成 h61 停顿原因。 | 未证明 |
| 驱动/CUDA 等待 | h61 GPU clock-event flags 全零，没有非法访问日志；dmesg 被拒绝。全零时钟原因不等于驱动已排除。 | 未证明 |
| 候选 graph 或输出错误 | 原停顿发生在 graph OFF baseline；候选从未运行。token/acceptance 并无错误。 | 不支持该归因 |

## h62：一次诊断运行及其停止条件

完全复用 h61 baseline 的数值配置、frozen oracle、串行/并发 correctness、c4/c8/c16 warmup 顺序；计划只跑第一窗口，不运行候选，总时限 1200 秒。

观测仅包括：原 CuTe compile 调用计时、stdlib GC callbacks、每秒进程 tick、/proc/cgroup 数据和异步 faulthandler 线程转储。观测数据写在本地 XFS，与原 FUSE 服务日志分离。没有改变 GC 阈值、冻结时机、kernel、采样或调度算法。

运行实际持续约 175.5 秒，于 23:12:47 scheduler_0（PID 32906）exit -11。客户端出现不完整 SSE payload。立即结束 owned 服务；没有进入 c16 稳定态，没有原 60 秒停顿的复现证据。

已保存的 139 个完成请求：71 个 correctness（包括 32 个 c16 短 burst）和 68 个 c4 warmup。独立复查全部 token、5 项 acceptance、cold-prefill 与零 retraction 通过；不能代表正在进行而未完成的请求。

- 已完成请求 E2E 最大 9608.75 ms，无完成请求超过 20 秒。
- 只记录到 2 次 CuTe compile：2.050 秒和 1.347 秒，均在启动/初始请求阶段。
- GC 最大 0.440 秒；进程 tick 最大间隔 1.635 秒；host 采样最大间隔 1.004 秒。
- 最后转储以 `Timeout (0:00:20)!` 开始，打印 LayerNorm Python frame 后出现 `File ???, line 0 in ???`，再截断于 `File`。

CPython 3.11 社区存在异步 faulthandler 转储中途崩溃的报告，近期也有 code-metadata 生命周期竞态报告。这与本次观测器尾部形态相似，**高度怀疑观测器污染，但没有 native core/backtrace，不能确认 SIGSEGV 的确切根因**。最后出现 LayerNorm/GEMM frame 并不证明这些算子有错误。

nsys 2025.6.3 的环境检查显示 perf_event_open/sampling trigger 不可用；已有 py-spy 0.4.2 的非阻塞读取也因 Permission denied 失败。没有绕过容器限制，没有因为这个问题再次升级工具。

## 停手与保留状态

- 本轮不重试 h62，不复用异步 faulthandler 观察器，不在无根因证据下改 LayerNorm/GEMM/GC/调度框架。
- 服务仓库仍为 `0577109f4e659b8ef2ef3f979f1fd7f079b25821`，原稳定 baseline tag `dspark-stable-fa4-fixed64-matrix128-passed-20261007`。
- main 仍为 `0c1f1eb64af939379ce47045c56aa8d2508c54df`，tracked clean。
- 8357 个原有未跟踪路径全部保留；NUL 路径列表 SHA256：`9077a67cf4e576a78b7750734c1e956defe85114739b9bb6eed68bc828708b7e`。
- post-integrity 对源码、数值 policy、包版本、模型 metadata、SPS、oracle、GPU/driver 身份检查通过。GPU 无运行计算进程，owned 30117 端口已关闭。
- 没有新的优化成功结论，没有 GitHub push。本轮仅封存负面诊断证据。

如果以后恢复工作，最小范围应先隔离测量环境：将所有热日志/请求记录放本地 XFS，结束后再归档网络盘，并取得允许的外部 native CPU 栈观测。这是环境假设验证，不是已验证的停顿修复或提速。没有这样的条件，不宜继续 GPU 重跑；更没有依据大改框架。

## 参考资料

- [AutoDL 官方文件存储说明](https://www.autodl.com/docs/fs/)：网络共享存储与本地盘的 IO 取舍。这里只用于环境风险判断，没有声称该页面证明本次 60 秒事件。
- [SGLang PR #25810](https://github.com/sgl-project/sglang/pull/25810)：DeepSeek V4 稀有 MHC bucket 的冷启动停顿与预热方案；模型/路径不同，不能直接移植为本项目结论。
- [SGLang issue #30596](https://github.com/sgl-project/sglang/issues/30596)：embedding 负载下 GC 暂停；本项目已存在启动 freeze_gc，且不是相同量级事件。
- [Nsight Systems 容器采样要求](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#collecting-data-within-a-container)：CPU 采样依赖宿主/容器权限；软件升级不能自动赋予权限。
- [CPython issue #116008](https://github.com/python/cpython/issues/116008)：Python 3.11 多线程异步栈转储中途 SIGSEGV。
- [CPython issue #158200](https://github.com/python/cpython/issues/158200) 与 [PR #158202](https://github.com/python/cpython/pull/158202)：近期 code metadata 生命周期导致的 faulthandler 竞态案例；其复现版本与本环境不同，只作为风险线索，不当作已确认的本机诊断。

原始 h62 数据、完整转储、observer 源码及只读核查在同目录证据包；其中 observer 已弃用，不应重新执行。
