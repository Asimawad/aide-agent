# AIDE Agents Guide

## New Modular Structure

Your codebase has been reorganized into a clean, modular structure:

```
aide/
├── agents/                    # ← NEW: All agent implementations
│   ├── __init__.py           # Exports all agents
│   ├── base.py               # Base Agent class (parent for all)
│   ├── baseline_agent.py     # Vanilla AIDE (no ITS)
│   ├── code_chain_agent.py   # Task Decomposition (Planner-Coder)
│   ├── planner_agent.py      # Planner-based agent
│   ├── self_consistency_agent.py  # Self-Consistency (SC)
│   └── self_debug_agent.py   # Self-Reflection (SR)
├── backend/                   # LLM backends (vLLM, OpenAI, etc.)
├── utils/                     # Utilities (config, prompts, metrics, etc.)
├── interpreter.py             # Code execution sandbox
├── journal.py                 # Solution tree tracking
└── run.py                     # Main entry point
```

---

## Available Agents

| Agent | ITS Strategy | Config Value | Best For | Paper Section |
|-------|--------------|--------------|----------|---------------|
| **BaselineAgent** | None (vanilla AIDE) | `baseline` | Establishing baseline performance | Baseline |
| **SelfDebugAgent** | Self-Reflection (SR) | `self-debug` or `self-reflection` | Mid-sized models (14B), fixing contained bugs | 4.3 |
| **SelfConsistencyAgent** | Self-Consistency (SC) | `self-consistency` | Improving reliability, 14B models | 4.4 |
| **CodeChainAgent** | Task Decomposition | `codechain`, `codechain_v2`, `codechain_v3` | Large models (32B), complex tasks | 4.5 |
| **PlannerAgent** | Planning-focused | `planner` | Separate planning phase | - |

---

## Quick Start

### 1. Verify Everything Works

```bash
cd /home/asim_aims_ac_za/aide-agent
python verify_agents.py
```

This will:
- ✓ Test all imports
- ✓ Load configuration
- ✓ Instantiate each agent
- ✓ Check example tasks and competition data

### 2. Run a Quick Test (3 steps)

**Baseline Agent:**
```bash
aide data_dir="aide/example_tasks/house_prices" \
     goal="Predict the sales price for each house" \
     eval="Use RMSE" \
     agent.ITS_Strategy="baseline" \
     agent.steps=3
```

**Self-Consistency (14B):**
```bash
aide data_dir="aide/example_tasks/spooky-author-identification" \
     goal="Predict the author" \
     agent.ITS_Strategy="self-consistency" \
     agent.selfConsistency.num_responses=3 \
     agent.steps=3
```

**Code-Chain (32B - Best Performance):**
```bash
aide data_dir="aide/example_tasks/leaf-classification" \
     goal="Classify leaf species" \
     agent.ITS_Strategy="codechain" \
     agent.steps=3
```

### 3. Run Full Benchmark (25 steps)

```bash
aide data_dir="data/aerial-cactus-identification" \
     goal="Identify images containing cacti" \
     agent.ITS_Strategy="codechain" \
     agent.steps=25 \
     wandb.enabled=true \
     wandb.project="MLE_BENCH_AIDE"
```

---

## Configuration (`aide/utils/config.yaml`)

### Key Settings

```yaml
# Agent strategy selection
agent:
  ITS_Strategy: "codechain"  # ← Change this!
  steps: 25
  
  # Model configuration
  code:
    model: "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic"
    planner_model: "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic"
    temp: 0.8
    max_new_tokens: 2048
  
  # Self-consistency settings
  selfConsistency:
    selection_strategy: "interpreter_first_success"  # or "interpreter_best_metric"
    num_responses: 3
  
  # Search policy
  search:
    num_drafts: 5
    debug_prob: 0.65
    max_debug_depth: 50

# Inference engine
inference_engine: vllm  # or "openai", "ollama", etc.
```

---

## Command-Line Overrides

You can override any config parameter:

```bash
aide data_dir="..." goal="..." \
     agent.ITS_Strategy="self-consistency" \
     agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
     agent.selfConsistency.num_responses=5 \
     agent.steps=10 \
     wandb.project="my-experiments"
```

---

## For Benchmarking

### Competition Selection

Your available competitions (in `data/`):
- `aerial-cactus-identification` - Image Classification
- `leaf-classification` - Image Classification
- `spooky-author-identification` - Text Classification
- `random-acts-of-pizza` - Text Classification
- `jigsaw-toxic-comment-classification-challenge` - Text Classification
- `text-normalization-challenge-english-language` - Seq2Seq
- `text-normalization-challenge-russian-language` - Seq2Seq

### Benchmark Script Template

```bash
#!/bin/bash
# benchmark_agent.sh

AGENT_TYPE="codechain"
MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic"
COMPETITIONS=(
    "aerial-cactus-identification"
    "leaf-classification"
    "spooky-author-identification"
)

for COMP in "${COMPETITIONS[@]}"; do
    echo "Running $AGENT_TYPE on $COMP..."
    aide data_dir="data/$COMP" \
         competition_name="$COMP" \
         agent.ITS_Strategy="$AGENT_TYPE" \
         agent.code.model="$MODEL" \
         agent.steps=25 \
         wandb.enabled=true \
         wandb.run_name="${AGENT_TYPE}_${COMP}"
done
```

---

## Troubleshooting

### Import Errors
```bash
# Re-install in development mode
pip install -e .
```

### vLLM Server Not Running
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic" \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85
```

### Missing OpenAI API Key
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Next Steps

1. **Run verification**: `python verify_agents.py`
2. **Quick test**: Run one agent for 3 steps
3. **Full benchmark**: Run all agents on your competition suite
4. **Analyze results**: Check `logs/` directory and W&B

---

**Reference**: See `aide.md` for full thesis documentation and research findings.

