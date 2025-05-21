#!/bin/bash
set -e
set -o pipefail

echo "Entrypoint script started."

export VLLM_TRACE_LEVEL=DEBUG

# --- Start first model ---
echo "Starting vLLM server #1 with RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic on port 8000..."
touch vllm_deepseek_14B.log

python -m vllm.entrypoints.openai.api_server \
    --model "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic" \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 50 \
    --trust-remote-code \
    --enforce-eager &> vllm_deepseek_14B.log &

VLLM_14B_PID=$!
echo "Started model 1 with PID: $VLLM_14B_PID"

# # --- Start second model ---
# echo "Starting vLLM server #2 with deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct on port 8001..."
# touch vllm_deepseek_coder_v2.log

# python -m vllm.entrypoints.openai.api_server \
#     --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
#     --port 8001 \
#     --device cuda \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.45 \
#     --max-num-batched-tokens 16384 \
#     --max-num-seqs 50 \
#     --trust-remote-code \
#     --enforce-eager &> vllm_deepseek_coder_v2.log &

# VLLM_CODER_V2_PID=$!
# echo "Started model 2 with PID: $VLLM_CODER_V2_PID"

# --- Wait for both servers to be healthy ---
timeout=1200
start=$(date +%s)

echo "Waiting for model 1 health on port 8000..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    now=$(date +%s)
    if (( now - start > timeout )); then
        echo "Timeout waiting for model 1"
        exit 1
    fi
    sleep 5
done
echo "Model 1 is healthy."

# echo "Waiting for model 2 health on port 8001..."
# while ! curl -s http://localhost:8001/health > /dev/null; do
#     now=$(date +%s)
#     if (( now - start > timeout )); then
#         echo "Timeout waiting for model 2"
#         exit 1
#     fi
#     sleep 5
# done
# echo "Model 2 is healthy."

# Keep script running or exec your command
wait








# # export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# # export MODEL_NAME="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
# # export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
# # export MODEL_NAME="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# # export MODEL_NAME="RedHatAI/DeepSeek-Coder-V2-Lite-Instr uct-FP8"

# set -e                          
# set -o pipefail               

# echo "Entrypoint script started."

# export VLLM_TRACE_LEVEL=DEBUG
# export OLLAMA_NUM_PARALLEL=8
# export OLLAMA_MAX_LOADED_MODELS=6
# export OLLAMA_MAX_QUEUE=1024


# # --- STEP 1: Start vLLM server using SYSTEM Python ---
# echo "Starting vLLM server using system python..."
# # Redirect stdout and stderr to a log file
# touch /home/vllm_server.log

# python -m vllm.entrypoints.openai.api_server \
#     --model "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"  \
#     --port 8000 \
#     --dtype bfloat16 \
#     --device cuda \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.85 \
#     --max-num-batched-tokens 32768 \
#     --max-num-seqs 256  \
#     --trust-remote-code   \
#     --enforce-eager  &> /home/vllm_server.log &

# while [ ! -f /home/vllm_server.log ]; do sleep 0.2; done
# VLLM_PID=$!

# tail -n +1 -f /home/vllm_server.log &
# TAIL_PID=$!
# echo "vLLM server started with PID: $VLLM_PID, logging to /home/vllm_server.log with tail PID: $TAIL_PID # "


# echo "Waiting for vLLM server on port  8000..."
# timeout_seconds=1200
# start_time=$(date +%s)

# while true; do
#     current_time=$(date +%s)
#     if [ $(($current_time - $start_time)) -ge $timeout_seconds ]; then
#         echo "vLLM server did not become healthy within $timeout_seconds seconds."
#         exit 1
#     fi
#     if curl -s http://localhost:8000/health > /dev/null; then
#         echo "vLLM server is healthy."
#         kill $TAIL_PID
#         break
#     fi
#     echo "vLLM server is not healthy yet, time passed -> $(($current_time - $start_time)) ."

#     sleep 20
# done



# echo "Starting Ollama server..."
# ollama serve &
# OLLAMA_PID=$!
# echo "Ollama server started with PID: $OLLAMA_PID"
# echo "Executing command: $@"
# exec "$@"
