#!/bin/bash
# Serve Kimi K2.6 (native INT4, about 595 GB of weights) for free_caption.py on one 8-GPU node: vLLM 0.30.0, TP8,
# an OpenAI-compatible endpoint at http://<host>:${PORT}/v1/chat/completions, model name "kimi".
#
#     bash serve_kimi.sh [MODEL]        MODEL: a Hugging Face id or local path (default: moonshotai/Kimi-K2.6)
#
# Environment: VLLM (vllm binary, default: vllm), PORT (default: 8000), LOG (server log, default: vllm_kimi.log),
# VLLM_CACHE_ROOT (optional, vLLM's compile cache; point it at a shared disk to reuse it across nodes).
# Refuses to start if any process is on the GPUs. Weights load with the "prefetch" strategy: each TP rank reads its
# share of the shards sequentially just before loading (on a network file system a page cache warmed minutes earlier
# can go cold, and lazy mmap loading then crawls).
set -u
MODEL=${1:-moonshotai/Kimi-K2.6}
VLLM=${VLLM:-vllm}
PORT=${PORT:-8000}
LOG=${LOG:-vllm_kimi.log}
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPUs busy"; exit 1; fi
setsid nohup "$VLLM" serve "$MODEL" \
  --tensor-parallel-size 8 --mm-encoder-tp-mode data --trust-remote-code \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --safetensors-load-strategy prefetch \
  --host 0.0.0.0 --port "$PORT" --max-model-len 65536 \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.9 \
  --served-model-name kimi > "$LOG" 2>&1 < /dev/null &
echo "launched pid $!"
