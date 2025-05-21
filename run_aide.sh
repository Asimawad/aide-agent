#!/usr/bin/env bash
set -e                          
set -o pipefail               

O4_MODEL="o4-mini-2025-04-16"
# PLANNER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
competition_name="spooky-author-identification"
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
    agent.ITS_Strategy="Baseline" \
    agent.code.model="${CODER_MODEL}" \
    agent.code.planner_model="${PLANNER_MODEL}" \
    agent.code.temp=0.6 \
    agent.code.max_new_tokens=2048 \
    agent.code.num_return_sequences=1 \
    agent.feedback.model="${FEEDBACK_MODEL}" \
    agent.search.max_debug_depth=5 \
    agent.search.debug_prob=0.7 \
    wandb.enabled=True \
    wandb.project="MLE_BENCH_AIDE_VM" \
    wandb.entity=asim_awad 
