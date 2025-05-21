#!/usr/bin/env bash
# chmod +x run_aide.sh
set -e                          
set -o pipefail               

# echo "Entrypoint script started."
# sudo touch vllm_server.log
# sudo chown $(whoami) vllm_server.log
# sudo chmod 777 vllm_server.log
# echo "vLLM server log file created at vllm_server.log"
# echo "Starting vLLM server..."
# export VLLM_LOG_LEVEL=DEBUG
# export VLLM_LOG_FILE="vllm_server.log"
# export VLLM_LOG_FORMAT="%(asctime)s %(levelname)s %(message)s"
# export VLLM_TRACE_LEVEL=DEBUG
# export MODEL_NAME="RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic"



# python -m vllm.entrypoints.openai.api_server \
#     --model "${MODEL_NAME}" \
#     --port 8000 \
#     --device cuda \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 16384 \
#     --trust-remote-code \
#     --max-num-batched-tokens 32000 \
#     --enforce-eager \
#     --max-num-seqs 256 &> vllm_server.log &
# while [ ! -f vllm_server.log ]; do sleep 0.2; done

# VLLM_PID=$!

# tail -n +1 -f vllm_server.log &
# TAIL_PID=$!
# echo "vLLM server started with PID: $VLLM_PID, logging to vllm_server.log with tail PID: $TAIL_PID # "


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

echo "Executing command: aide"

DEEPSEEK_ID="deepseek/deepseek-coder" #"deepseek-ai/DeepSeek-Coder-V2-16B" 
# O4_MODEL="o4-mini-2025-04-16"
# CODER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# CODER_MODEL="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8"
# PLANNER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
PLANNER_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
competition_name="spooky-author-identification"
# competition_name="spooky-author-identification"
FEEDBACK_MODEL="o4-mini-2025-04-16"
data_dir="aide/example_tasks/spooky-author-identification"
GOAL="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley"
EVAL="Use multi-class logarithmic loss between predicted author probabilities and the true label." 
aide \
    data_dir="${data_dir}/" \
    goal="${GOAL}" \
    eval="${EVAL}" \
    log_level="INFO" \
    competition_name="${competition_name}" \
    agent.steps=25 \
    agent.time_limit=7200 \
    agent.obfuscate=False \
    agent.ITS_Strategy="planner" \
    agent.code.model="${FEEDBACK_MODEL}" \
    agent.code.planner_model="${FEEDBACK_MODEL}" \
    agent.code.temp=0.6 \
    agent.code.max_new_tokens=2048 \
    agent.code.num_return_sequences=1 \
    agent.feedback.model="${FEEDBACK_MODEL}" \
    agent.search.max_debug_depth=5 \
    agent.search.debug_prob=0.7 \
    wandb.enabled=True \
    wandb.project="MLE_BENCH_AIDE_VM" \
    wandb.entity=asim_awad 
