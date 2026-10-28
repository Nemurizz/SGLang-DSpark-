#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/src/sglang-main
export PYTHONPATH=/root/autodl-tmp/src/sglang-main/python
export SGLANG_RAGGED_VERIFY_MODE=compact
export SGLANG_SIMULATE_ACC_LEN=-1
export SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,verify_path,reqs
export SGLANG_DSPARK_ENABLE_SPS_RECORD=0
export SGLANG_DSPARK_PREFIX_CACHE=0
export SGLANG_DSPARK_PREFIX_CACHE_SHADOW=0
export SGLANG_DSPARK_DISABLE_RADIX_CACHE=1
export SGLANG_DSPARK_RAGGED_VERIFY_GRAPH=0
export SGLANG_DSPARK_COMPACT_SCAN_CUDA=0
export SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH=0
export SGLANG_DSPARK_FOLDED_PROPOSAL=1
export SGLANG_DSPARK_FOLDED_SAMPLING=0
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_COMPACT_MIN_TOKEN_SAVINGS_RATIO=0
exec /root/conda-envs/sglang/bin/python -m sglang.launch_server \
  --model-path /root/autodl-tmp/models/Qwen3-14B \
  --tokenizer-path /root/autodl-tmp/models/Qwen3-14B \
  --trust-remote-code --host 127.0.0.1 --port 30217 \
  --mem-fraction-static 0.70 --max-running-requests 16 --max-total-tokens 131072 \
  --random-seed 20261029 --attention-backend fa4 --decode-attention-backend fa4 \
  --prefill-attention-backend fa4 --sampling-backend pytorch --stream-interval 1 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /root/autodl-tmp/models/dspark_qwen3_14b_block7 \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
  --speculative-dspark-block-size 7 --speculative-draft-attention-backend fa4 \
  --speculative-dsa-topk-backend sgl-kernel \
  --speculative-dspark-sps-table-path /root/autodl-tmp/results/sps-probes/dspark14-sps-probe-20260830-210115/qwen3-14b-rtxpro6000-additive-sps.json \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --bf16-gemm-backend torch \
  --page-size 1 --disable-cuda-graph --disable-radix-cache \
  --enable-deterministic-inference
