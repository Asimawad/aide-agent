#!/usr/bin/env bash
# set -e                          
# set -o pipefail   
# cp -r logs/ logs_old/
# rm -rf workspaces/ wandb/ logs/
# mkdir -p logs/
CODER_MODEL="o4-mini-2025-04-16"
# O4_MODEL="o3-mini"
# DeepSeek_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic"
# CODER_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic"
# CODER_MODEL="o3-mini"
# ${O4_MODEL} #"RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"

# PLANNER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
competition_name="leaf-classification"
data_dir="data/leaf-classification"
GOAL="Predict the leaf species" 
EVAL="Use the accuracy score metric between the predicted and observed values." 
aide \
    data_dir="${data_dir}/" \
    desc_file="${data_dir}/leaf-classification.md" \
    log_level="DEBUG" \
    competition_name="${competition_name}" \
    agent.steps=5 \
    agent.time_limit=36000 \
    agent.obfuscate=False \
    agent.ITS_Strategy="self_consistency" \
    agent.code.model="${CODER_MODEL}" \
    agent.code.planner_model="${CODER_MODEL}" \
    agent.code.temp=0.8 \
    agent.code.max_new_tokens=4096 \
    agent.code.num_return_sequences=1 \
    agent.feedback.model="${CODER_MODEL}" \
    agent.search.max_debug_depth=50 \
    agent.search.debug_prob=0.8 \
    agent.search.num_drafts=1 \
    wandb.enabled=True \
    wandb.project="MLE_BENCH_AIDE_VM" \
    wandb.entity=asim_awad
    

