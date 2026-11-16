# DSpark 两个组合故障修复记录（2026-11-17）

本记录只覆盖本轮要求的两个故障：adaptive draft 的 planner cost vector 越界，以及 FP8 KV cache 与 FA4 attention 的 dtype 不匹配。修复已经写入远端 SGLang 工作树，并在 RTX PRO 6000 Blackwell + Qwen3-14B + DSpark block7 上做了运行时验证。

## 1. Adaptive draft：`7 < 13`

### 现象

开启 adaptive draft 后，服务在 planner 选 budget 时退出：

```text
ValueError: cached additive step-time vector is shorter than the candidate pool: 7 < 13
```

这不是模型输出错误，而是 planner 的缓存生命周期错误。旧代码只按 `num_requests` 缓存 additive step-time 向量。adaptive 路径第一次调用 planner 时，候选池可能较短；后续 adaptive 重新计算最大 verify 长度后，候选池变长，但仍复用了第一次生成的短向量。

### 修复

远端文件：`python/sglang/srt/speculative/dspark_components/dspark_planner.py`

`HostConfidenceBudgetPlanner._cached_additive_step_time()` 现在每次按当前配置重新计算：

```python
required_budgets = (
    num_requests * self.cfg.resolved_max_verify_len() + 1
)
cached = self._additive_step_time_cache.get(num_requests)
if cached is None or cached.numel() < required_budgets:
    cached = _additive_step_time_tensor(
        table=self.sps_table,
        num_requests=num_requests,
        num_budgets=required_budgets,
    )
    self._additive_step_time_cache[num_requests] = cached
```

关键点是缓存命中后仍然检查长度。这样不会把 adaptive 的候选池变化误认为同一个固定形状的 planner 输入。

### 验证结果

- `py_compile`：通过。
- 直接 planner 单测：预置长度 7、当前需要 14 时，缓存自动扩容到 14，通过。
- GPU 服务冒烟：通过；8 个请求全部完成，cached tokens 共 3529，没有再次出现 `7 < 13`。
- 冒烟服务保持 deterministic + FA4 + BF16 KV，说明本修复没有依赖 KV 或 backend 绕过。

这次只证明“adaptive 路径不再因缓存向量长度崩溃”，没有把 8 请求冒烟结果包装成 planner 的端到端加速结论。后续若要测收益，应在修复后的路径上重新做随机 ABBA、多轮 planner CPU、budget、GPU step、TTFT、TPOT、E2E 和 p95 配对实验。

## 2. FP8 KV：`BF16 Q != FP8 K/V`

### 现象

开启 `--kv-cache-dtype fp8_e4m3` 并使用 FA4 时，首个请求在 attention kernel 内退出：

```text
AssertionError: q.dtype torch.bfloat16 != k.dtype torch.float8_e4m3fn
```

原因是 target 的 query 仍按 BF16 计算，而 FA4 直接读取 paged K/V；这条 FA4 路径要求 Q、K、V 的计算 dtype 一致，它不会替 FP8 KV 做本轮需要的 dequant。问题发生在 attention backend 组合层，不是“FP8 量化没有收益”。

### 修复

远端文件：`python/sglang/srt/server_args.py`

在 attention backend 解析完成后增加 `_handle_fp8_kv_cache_compatibility()`：

1. 只有 `kv_cache_dtype == fp8_e4m3` 且实际 prefill/decode backend 含 `fa4` 时触发。
2. 若同时请求 deterministic inference，立即给出可操作的错误，避免服务启动后才在 kernel 中崩溃：

```text
FP8 KV cache is incompatible with FA4 under deterministic inference ...
Disable --enable-deterministic-inference and let SGLang select FlashInfer,
or use --kv-cache-dtype bf16.
```

3. 非 deterministic 场景下，只把实际使用 FA4 的 backend slot 改为 `flashinfer`。FlashInfer 负责 FP8 K/V 的 dequant，再以 BF16 完成 attention；radix/paired cache 保持开启。
4. 没有改变 BF16 KV、非 FA4 backend、或普通 deterministic BF16 服务的路径。

这保留了 FP8 KV 的显存目的，同时避免把 FP8 K/V 硬塞给不兼容的 FA4 kernel。它不是把 Q 强行转成 FP8，也不是吞掉 dtype assertion。

### 验证结果

- `py_compile server_args.py`：通过。
- 源码自动路由冒烟：命令仍传 `--attention-backend fa4 --prefill-attention-backend fa4 --decode-attention-backend fa4 --kv-cache-dtype fp8_e4m3`，没有在脚本中手工替换 backend；服务最终报告：
  - `kv_cache_dtype = fp8_e4m3`
  - `attention_backend = flashinfer`
  - `prefill_attention_backend = flashinfer`
  - `decode_attention_backend = flashinfer`
  - `disable_radix_cache = false`
  - `enable_deterministic_inference = false`
- 8 个请求全部完成，cached tokens 共 3529；FP8 KV 运行时不再出现 Q/K dtype assertion。
- deterministic + FA4 + FP8 KV 的回归启动会在 `ServerArgs.resolve_once()` 阶段明确失败，错误信息指向两个可行选择，而不是进入 kernel 后崩溃。

本冒烟只验证兼容性、缓存命中和服务可用性，不比较 FlashInfer 与 FA4 的性能，也不宣称 FP8 KV 的质量结果。生产上线前仍应补做 BF16 KV 对照、FP8 KV 质量集、长上下文、高并发和尾延迟测试。

## 3. 修改文件与复现材料

- adaptive planner：`python/sglang/srt/speculative/dspark_components/dspark_planner.py`
- FP8 KV/backend 兼容：`python/sglang/srt/server_args.py`
- DSpark 可移植补丁中的 adaptive 片段：`sglang-dspark-dynamic-verification.patch`
- adaptive 冒烟输出：`/root/autodl-fs/adaptive-smoke-after-fix-20261117/result.json`
- FP8 KV 源码自动路由输出：`/root/autodl-fs/kv-quant-smoke-source-route-20261117/result.json`
- deterministic guard 日志：`/root/autodl-fs/kv-quant-deterministic-guard-20261117/server.log`

远端工作树的修改是运行时源码修复；本仓库保存的是修复说明、可移植 DSpark patch 和实验协议。两项修复都没有证明“打开所有开关一定更快”，它们解决的是两个确定的可用性/正确性阻断点。
