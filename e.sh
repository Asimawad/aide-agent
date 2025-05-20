#!/bin/bash
set -e                          
set -o pipefail               

echo "Entrypoint script started."

export VLLM_TRACE_LEVEL=DEBUG

# export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# export MODEL_NAME="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
# export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
# export MODEL_NAME="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
export MODEL_NAME="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8"
python -m vllm.entrypoints.openai.api_server \
    --model "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8" \
    --port 8000 \
    --device cuda \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --enforce-eager \
    --max-num-seqs 256 &> /home/vllm_server.log &
while [ ! -f /home/vllm_server.log ]; do sleep 0.2; done
VLLM_PID=$!

tail -n +1 -f /home/vllm_server.log &
TAIL_PID=$!
echo "vLLM server started with PID: $VLLM_PID, logging to /home/vllm_server.log with tail PID: $TAIL_PID # "


echo "Waiting for vLLM server on port  8000..."
timeout_seconds=1200
start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    if [ $(($current_time - $start_time)) -ge $timeout_seconds ]; then
        echo "vLLM server did not become healthy within $timeout_seconds seconds."
        exit 1
    fi
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "vLLM server is healthy."
        kill $TAIL_PID
        break
    fi
    echo "vLLM server is not healthy yet, time passed -> $(($current_time - $start_time)) ."

    sleep 20
done

echo "Executing command: $@"
exec "$@"

python -m vllm.entrypoints.openai.api_server \
       --model  RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8 \
       --trust-remote-code \
       --port 8000 \
       --max-model-len 4096 \   
       --gpu-memory-utilization 0.85  