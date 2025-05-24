#!/usr/bin/env bash
# set -e                          
# set -o pipefail   
cp -r logs/ logs_old/
rm -rf workspaces/ wandb/ logs/
mkdir -p logs/
# first check if the data/ is present
if [ ! -d "data/" ]; then
    echo "Error: data/ directory not found. downloading it ...."
    python aide/utils/drive_download.py
fi

O4_MODEL="o4-mini-2025-04-16"
DeepSeek_MODEL="gpt-4-turbo"
CODER_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
# CODER_MODEL="o3-mini"
# ${O4_MODEL} #"RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"

PLANNER_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
competition_name="aerial-cactus-identification"
data_dir="data/aerial-cactus-identification"
GOAL="Predict the class of a cactus" 
EVAL="Use the accuracy score metric between the predicted and observed values." 
aide \
    data_dir="${data_dir}/" \
    desc_file="${data_dir}/aerial-cactus-identification.md" \
    log_level="DEBUG" \
    competition_name="${competition_name}" \
    agent.steps=5 \
    agent.time_limit=36000 \
    agent.obfuscate=False \
    agent.ITS_Strategy="planner" \
    agent.code.model="${PLANNER_MODEL}" \
    agent.code.planner_model="${PLANNER_MODEL}" \
    agent.code.temp=0.8 \
    agent.code.max_new_tokens=4096 \
    agent.code.num_return_sequences=1 \
    agent.feedback.model="${O4_MODEL}" \
    agent.search.max_debug_depth=5 \
    agent.search.debug_prob=0.65 \
    wandb.enabled=True \
    wandb.project="MLE_BENCH_AIDE_VM" \
    wandb.entity=asim_awad
    

