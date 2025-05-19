"""

aide data_dir="data/spooky-author-identification" \
    goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" \
    eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." \
    agent.code.model=o3-mini \
    agent.ITS_Strategy="none" \
    agent.steps=3 \
    inference_engine="vllm" \
    agent.code.planner_model="Qwen/Qwen2-0.5B-Instruct" \
    competition_name=spooky-author-identification



"""