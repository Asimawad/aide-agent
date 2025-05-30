#!/bin/bash
set -e                          
set -o pipefail               

echo "Entrypoint script started."

export VLLM_TRACE_LEVEL=DEBUG

# --- Start first model ---
echo "Starting vLLM server #1 with RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic on port 8000..."
touch ./vllm_deepseek_7B.log

.aide-ds/bin/python -m vllm.entrypoints.openai.api_server \
    --model "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic" \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda \
    --max-model-len 19000 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 10 \
    --trust-remote-code \
    --enforce-eager &> ./vllm_deepseek_7B.log &

VLLM_7B_PID=$!
echo "Started model with PID: $VLLM_7B_PID"

timeout=1200
start=$(date +%s)

echo "Waiting for model health on port 8000..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    now=$(date +%s)
    if (( now - start > timeout )); then
        echo "Timeout waiting for model 1"
        exit 1
    fi
    echo "Model not healthy yet, waiting..."
    sleep 5
done
echo "Model is healthy."



echo "Executing command: $@"
exec "$@"