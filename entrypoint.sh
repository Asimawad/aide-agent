#!/bin/bash
set -e
set -o pipefail

echo "Entrypoint script started..."
mkdir -p ./logs
touch ./logs/vllm_coder.log ./logs/vllm_planner.log
sudo chown $(whoami) ./logs/vllm_coder.log
sudo chown $(whoami) ./logs/vllm_planner.log
sudo chmod 777 ./logs/vllm_coder.log
sudo chmod 777 ./logs/vllm_planner.log
echo "vLLM server log file created at ./logs/vllm_server.log"
# echo "Starting vLLM server..."
export VLLM_LOG_LEVEL=DEBUG
export VLLM_LOG_FILE="./logs/vllm_server.log"
export VLLM_LOG_FORMAT="%(asctime)s %(message)s"
export VLLM_TRACE_LEVEL=DEBUG
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_QUEUE=1024
export first_model_log="./logs/vllm_coder.log"
export second_model_log="./logs/vllm_planner.log"

export FEEDBACK_MODEL="o4-mini-2025-04-16"
export CODER_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
# export PLANNER_MODEL="deepseek-ai/DeepSeek-Coder-V2-16B"

# --- Start first model ---
echo "Starting vLLM server for coder model #1 with $CODER_MODEL on port 8000..."
touch $first_model_log
if [ -n "$CODER_MODEL" ]; then
# 
    python -m vllm.entrypoints.openai.api_server \
        --model "$CODER_MODEL" \
        --port 8000 \
        --dtype bfloat16 \
        --device cuda \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.9 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 3 \
        --trust-remote-code \
        --enforce-eager &> $first_model_log &

    CODER_MODEL_PID=$!
    echo "Started model 1 with PID: $CODER_MODEL_PID"

    tail -n +1 -f $first_model_log &
    TAIL_PID=$!

    # Wait for the first model to be healthy
    timeout=1200
    start=$(date +%s)

    echo "Waiting for model $CODER_MODEL health on port 8000..."
    while ! curl -s http://localhost:8000/health > /dev/null; do
        now=$(date +%s)
        if (( now - start > timeout )); then
            echo "Timeout waiting for model 1"
            exit 1
        fi
        sleep 5
    done
    kill $TAIL_PID
    echo "Model $CODER_MODEL is healthy."
else
    echo "CODER_MODEL is not set. Skipping first model start."
fi

if [ -n "$PLANNER_MODEL" ]; then
    echo "PLANNER_MODEL is set to $PLANNER_MODEL"
    # # --- Start second model ---
    echo "Starting vLLM server for planner model #2 with $PLANNER_MODEL on port 8001..."
    touch $second_model_log

    python -m vllm.entrypoints.openai.api_server \
        --model "$PLANNER_MODEL" \
        --port 8001 \
        --device cuda \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.45 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 50 \
        --trust-remote-code \
        --enforce-eager &> $second_model_log &

    PLANNER_PID=$!
    echo "Started model 2 with PID: $PLANNER_PID"
    tail -n +1 -f $second_model_log &
    TAIL_PID=$!
    # Wait for the second model to be healthy
    timeout=1200
    start=$(date +%s)
    echo "Waiting for model 2 health on port 8001..."
    while ! curl -s http://localhost:8001/health > /dev/null; do
        now=$(date +%s)
        if (( now - start > timeout )); then
            echo "Timeout waiting for model 2"
            exit 1
        fi
        sleep 5
    done
    echo "Model 2 is healthy."
    echo "Started model 2 with PID: $PLANNER_PID"
else
    echo "PLANNER_MODEL is not set. Skipping second model start."
fi


# echo "Starting Ollama server..."
# ollama serve &
# OLLAMA_PID=$!
# echo "Ollama server started with PID: $OLLAMA_PID"
chmod +x run_aide.sh

echo "Executing command: $@"
exec "$@"