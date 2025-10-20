# AIDE: AI-Driven Exploration for ML Engineering
### An Open-Source Agentic Framework for Autonomous Problem-Solving

AIDE (AI-Driven Exploration) is an open-source, autonomous agent designed to tackle end-to-end machine learning engineering tasks. It frames the complex, iterative process of ML development as a tree search through the space of possible code solutions. Powered by Large Language Models (LLMs), AIDE can draft initial solutions, debug faulty code, and iteratively improve upon working scripts to enhance performance, mirroring the workflow of a human data scientist.

This repository contains the implementation of the AIDE agent, along with several advanced **Inference-Time Scaling (ITS)** strategies designed to enhance the performance of smaller, open-source LLMs, making them competitive with large, proprietary models on challenging benchmarks like [MLE-Bench](https://github.com/openai/mle-bench).

![Solution Tree Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)

---

## Features
- **Agentic Tree Search:** Models ML engineering as a tree search, intelligently navigating through drafting, debugging, and improvement steps.
- **Local LLM Integration:** Comes with a high-throughput backend powered by [vLLM](https://github.com/vllm-project/vllm) for serving local, open-source models efficiently.
- **Plug-and-Play ITS Strategies:** Easily switch between different reasoning strategies via a simple configuration flag to find the best approach for your model and task.
- **Supported Strategies:** Includes implementations for `Self-Reflection`, `Planner-Coder` (Task Decomposition), `Self-Consistency`, and more.
- **Benchmark Ready:** Designed for rigorous evaluation on benchmarks like MLE-Bench.

---

## Quickstart

### 1. Setup Environment
Ensure you have Python >= 3.11 and `uv` installed.

```bash
# Clone the repository
git clone https://github.com/Asimawad/aide-agent.git
cd aide-agent

# Create and activate a virtual environment
uv venv .aide-ds --python 3.11
source .aide-ds/bin/activate

# Install dependencies (including PyTorch for CUDA 12.1)
uv pip install  --index-strategy unsafe-best-match  --extra-index-url https://download.pytorch.org/whl/cu124 -e .


uv pip install torch~=2.6.0
uv pip install --prerelease=allow torch-xla[tpu]~=2.6.0 
-f https://storage.googleapis.com/libtpu-releases/index.html
# Set your OpenAI API Key (used for the reliable feedback/judge model)
export OPENAI_API_KEY="<your-openai-api-key>"```

### 2. Launch the Local LLM Server
AIDE works best with a locally served open-source model for code generation. We use `vLLM` for high-performance inference.

In a separate terminal, launch the vLLM server with your chosen model. For example, to serve the DeepSeek 14B model:
```bash
# Make sure your environment is activated: source .aide-ds/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code
```

### 3. Run Your First AIDE Experiment
Now, in your original terminal, you can run an AIDE experiment.

**Example: House Price Prediction Task**
```bash
aide data_dir="aide/example_tasks/house_prices" \
     goal="Predict the sales price for each house" \
     eval="Use the RMSE metric between the logarithm of the predicted and observed values." \
     agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
     agent.steps=25
```
AIDE will now start the run. You can monitor its progress in the console and find the results in the `runs/` and `workspaces/` directories upon completion.

---

## How AIDE Works
AIDE's problem-solving approach is centered around a **Solution Space Tree Search**. This process has three main components:

1.  **The Solution Generator (The LLM):** Proposes new solutions by either creating novel drafts or making changes to existing solutions by fixing bugs or introducing improvements.
2.  **The Evaluator:** Assesses the quality of each proposed solution by executing the code in a sandboxed environment and parsing the output (tracebacks, printed metrics) to determine if the solution is buggy and what its performance score is.
3.  **The Search Policy:** A simple set of heuristics that selects the most promising node from the solution tree to serve as the base for the next iteration of refinement.

By repeatedly applying these steps, AIDE navigates the vast space of possible solutions, progressively refining its approach until it converges on an optimal solution.

## Using Inference-Time Scaling (ITS) Strategies

The true power of this framework lies in its ability to apply different reasoning strategies to the LLM. You can activate these via a single command-line flag.

### Self-Reflection (SR)
**What it does:** After a code execution fails, the agent is forced to first critique its own code and then revise it based on that critique. This is excellent for fixing contained bugs.

**How to run:**
```bash
aide data_dir="..." goal="..." \
     agent.ITS_Strategy="self-reflection" \
     agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
```

### Decomposed Task Generation (Planner-Coder / "DG")
**What it does:** This strategy separates the task into two phases: a "Planner" LLM creates a detailed, high-level plan, and then a "Coder" LLM implements that plan segment by segment. This is our best-performing strategy for high-capability models.

**How to run:**
```bash
aide data_dir="..." goal="..." \
     agent.ITS_Strategy="codechain" \
     agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
     agent.code.planner_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # You can use a different model for planning
```
*Note: The `codechain_v2` (per-segment reflection) and `codechain_v3` (chunked reflection) variants can also be set via the `ITS_Strategy` flag.*

### Self-Consistency (SC)
**What it does:** Generates *N* different solutions in parallel for the same prompt and then uses execution feedback to select the best one. This improves robustness and the chance of finding a working solution.

**How to run:**
```bash
aide data_dir="..." goal="..." \
     agent.ITS_Strategy="self-consistency" \
     agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
     agent.selfConsistency.num_responses=3 \
     agent.selfConsistency.selection_strategy="interpreter_first_success"
```

---

## Advanced Configuration
You can override any parameter from the command line. Key options include:
- `agent.steps=...`: Number of iterations for the agent (default: 25).
- `agent.search.num_drafts=...`: Number of initial solutions to explore (default: 5).
- `agent.code.temp=...`: The sampling temperature for the coding model (higher values increase creativity/randomness).
- `wandb.project=...`: To log your experiment results to Weights & Biases.

For a full list of configurable parameters, see the `aide/utils/config.yaml` file.

## Using AIDE as a Python Library
You can also integrate AIDE directly into your Python projects.

```python
import aide

# Initialize the experiment
exp = aide.Experiment(
    data_dir="aide/example_tasks/spooky-author-identification",
    goal="Predict the author of a sentence (Poe, Lovecraft, or Shelley).",
    eval="Use multi-class logarithmic loss."
)

# Configure the agent programmatically (optional)
exp.cfg.agent.ITS_Strategy = "self-consistency"
exp.cfg.agent.code.model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
exp.cfg.agent.selfConsistency.num_responses = 3

# Run the agent for 15 steps
best_solution = exp.run(steps=15)

print(f"Best solution's validation metric: {best_solution.valid_metric}")
print("--- Best Solution Code ---")
print(best_solution.code)
```

## Development
To install AIDE for development:
```bash
git clone https://github.com/Asimawad/aide-agent.git
cd aide-agent
uv pip install -e .
```






