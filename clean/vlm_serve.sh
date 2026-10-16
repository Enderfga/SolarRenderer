#!/bin/bash
# vlm_serve.sh MODEL_DIR : serve Qwen3.5-397B-A17B-FP8 with vLLM (tensor parallel over 8 GPUs) for vlm.py / vlm2.py.
# Waits up to 2 min for the GPUs to be free and refuses to start otherwise.
# env: VLLM (vllm executable, default "vllm"), PORT (8000), VLM_MODEL (served model name, default qwen3.5),
#      VLLM_LOG (default <system temp>/vllm_qwen3.5.log)
[ $# -ge 1 ] || { echo "usage: $0 MODEL_DIR"; exit 1; }
for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPUs still busy"; exit 1; fi
LOG=${VLLM_LOG:-${TMPDIR:-/tmp}/vllm_qwen3.5.log}
setsid nohup "${VLLM:-vllm}" serve "$1" \
  --tensor-parallel-size 8 --host 0.0.0.0 --port "${PORT:-8000}" --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":4,"video":0}' --allowed-local-media-path / \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.9 --max-num-seqs 128 \
  --served-model-name "${VLM_MODEL:-qwen3.5}" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! (log $LOG)"
