#!/usr/bin/env bash
# set -e                          
# set -o pipefail   
# cp -r logs/ logs_old/
# rm -rf workspaces/ wandb/ logs/
# mkdir -p logs/
O4_MODEL="o4-mini-2025-04-16"
DeepSeek_MODEL="gpt-4-turbo"
CODER_MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"
# CODER_MODEL="o3-mini"
# ${O4_MODEL} #"RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"

# PLANNER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
competition_name="text-normalization-challenge-english-language"
data_dir="data/text-normalization-challenge-english-language/"
GOAL="Predict the toxicity of a comment" 
EVAL="Use the F1 score metric between the predicted and observed values." 
aide \
    data_dir="${data_dir}/" \
    desc_file="${data_dir}/description.md" \
    log_level="DEBUG" \
    competition_name="${competition_name}" \
    agent.steps=25 \
    agent.time_limit=36000 \
    agent.obfuscate=False \
    agent.ITS_Strategy="Baseline" \
    agent.code.model="${DeepSeek_MODEL}" \
    agent.code.planner_model="${DeepSeek_MODEL}" \
    agent.code.temp=0.8 \
    agent.code.max_new_tokens=4096 \
    agent.code.num_return_sequences=1 \
    agent.feedback.model="${O4_MODEL}" \
    agent.search.max_debug_depth=50 \
    agent.search.debug_prob=0.65 \
    wandb.enabled=True \
    wandb.project="MLE_BENCH_AIDE_VM" \
    wandb.entity=asim_awad
    

