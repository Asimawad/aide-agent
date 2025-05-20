#!/usr/bin/env bash
# chmod +x run_aide.sh

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

echo "Executing command: aide"


CODER_MODEL="o4-mini-2025-04-16"
# CODER_MODEL="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8"
planner_model="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8"
competition_name="spooky-author-identification"
# competition_name="spooky-author-identification"
FEEDBACK_MODEL="o4-mini-2025-04-16"
data_dir="aide/example_tasks/spooky-author-identification"
GOAL="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley"
EVAL="Use multi-class logarithmic loss between predicted author probabilities and the true label." 
python aide \
        data_dir="${data_dir}/" \
        goal="${GOAL}" \
        eval=null \
        log_level="DEBUG" \
        competition_name="${competition_name}" \
        agent.steps=25 \
        agent.time_limit=7200 \
        agent.obfuscate=False \
        agent.ITS_Strategy="Baseline" \
        agent.code.model="${CODER_MODEL}" \
        agent.code.planner_model="" \
        agent.code.temp=0.6 \
        agent.code.max_new_tokens=2048 \
        agent.code.num_return_sequences=1 \
        agent.feedback.model="${FEEDBACK_MODEL}" \
        agent.search.max_debug_depth=5 \
        agent.search.debug_prob=0.7 \
        wandb.enabled=True \
        wandb.project="MLE_BENCH_AIDE" \
        wandb.entity=asim_awad \



