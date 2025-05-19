# AIDE: the Machine Learning CodeGen Agent

# How to use AIDE?
## Setup

Make sure you have uv and `Python>=3.11` installed and run:
```bash
uv venv .aide-ds --python 3.11 
source .aide-ds/bin/activate
uv pip install  --index-strategy unsafe-best-match  --extra-index-url https://download.pytorch.org/whl/cu124 -e .

```
Also install `unzip` to allow the agent to autonomously extract your data.

Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=<your API key>
```

## Running AIDE via the command line

To run AIDE:

```bash
aide data_dir="<path to your data directory>" goal="<describe the agent's goal for your task>" eval="<(optional) describe the evaluation metric the agent should use>"
```

For example, to run AIDE on the example [house price prediction task](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):

```bash
aide data_dir="example_tasks/house_prices" goal="Predict the sales price for each house" eval="Use the RMSE metric between the logarithm of the predicted and observed values." agent.code.model=deepseek-r1:latest wandb.project="my-aide-experiments"
```

### And here’s your model-friendly prompt-style instruction:

```bash
aide data_dir="data/spooky-author-identification" goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." agent.code.model=o3-mini agent.ITS_Strategy="none" agent.steps=3



python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2-0.5B-Instruct"  \
    --port 8001 \
    --dtype bfloat16 \
    --device cuda \
    --max-model-len 2000 \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code \
    --enforce-eager 

aide data_dir="data/spooky-author-identification" goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." agent.code.model=o3-mini agent.ITS_Strategy="self-reflection" agent.steps=10 competition_name="spooky-author-identification" agent.code.planner_model="Qwen/Qwen2-0.5B-Instruct"
### To use vllm for inference
.aide-ds/bin/python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda &


aide data_dir="aide/example_tasks/spooky-author-identification" goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." agent.code.model=deepseek-r1 agent.steps=2

aide data_dir="example_tasks/spooky-author-identification" goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." agent.code.model=deepseek-r1:latest wandb.project="my-aide-experiments"
```



## Arial-Cactus
# Ensure you are in the directory ABOVE 'aerial-cactus-identification'
# OR adjust data_dir accordingly if running from elsewhere.

# Activate your environment first, e.g.: source .aide-ds/bin/activate
```bash
aide data_dir="data/aerial-cactus-identification" \
     goal="Create a classifier capable of predicting whether an aerial image contains a cactus" \
     eval="Area under the ROC curve (AUC)" \
     agent.code.model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
     agent.steps=20 \
     agent.code.max_new_tokens=4096 \
     agent.code.temp=0.6 \
     wandb.project="aide-cactus-identification" 
```
Options:

- `data_dir` (required): a directory containing all the data relevant for your task (`.csv` files, images, etc.).
- `goal`: describe what you want the models to predict in your task, for example, "Build a timeseries forcasting model for bitcoin close price" or "Predict sales price for houses".
- `eval`: the evaluation metric used to evaluate the ML models for the task (e.g., accuracy, F1, Root-Mean-Squared-Error, etc.)

Alternatively, you can provide the entire task description as a `desc_str` string, or write it in a plaintext file and pass its path as `desc_file` ([example file](aide/example_tasks/house_prices.md)).

```bash
aide data_dir="my_data_dir" desc_file="my_task_description.txt"
```

The result of the run will be stored in the `logs` directory.

- `logs/<experiment-id>/best_solution.py`: Python code of _best solution_ according to the validation metric
- `logs/<experiment-id>/journal.json`: a JSON file containing the metadata of the experiment runs, including all the code generated in intermediate steps, plan, evaluation results, etc.
- `logs/<experiment-id>/tree_plot.html`: you can open it in your browser. It contains visualization of solution tree, which details the experimentation process of finding and optimizing ML code. You can explore and interact with the tree visualization to view what plan and code AIDE comes up with in each step.

The `workspaces` directory will contain all the files and data that the agent generated.

### Advanced Usage

To further customize the behaviour of AIDE, some useful options might be:

- `agent.code.model=...` to configure which model the agent should use for coding (default is `gpt-4-turbo`)
- `agent.steps=...` to configure how many improvement iterations the agent should run (default is 20)
- `agent.search.num_drafts=...` to configure the number of initial drafts the agent should generate (default is 5)

You can check the [`config.yaml`](aide/utils/config.yaml) file for more options.

## Using AIDE in Python

Using AIDE within your Python script/project is easy. Follow the setup steps above, and then create an AIDE experiment like below and start running:

```python
import aide
exp = aide.Experiment(
    data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
    goal="Build a timeseries forcasting model for bitcoin close price.",  # replace with your own goal description
    eval="RMSLE"  # replace with your own evaluation metric
)

best_solution = exp.run(steps=10)

print(f"Best solution has validation metric: {best_solution.valid_metric}")
print(f"Best solution code: {best_solution.code}")
```

## Development

To install AIDE for development, clone this repository and install it locally.
first, Install dependencies either using uv or pip
```bash
uv sync
uv pip install -e .
```
then:

```bash
git clone https://github.com/WecoAI/aideml.git
cd aideml
pip install -e .
```

Contribution guide will be available soon.

## Algorithm Description

AIDE's problem-solving approach is inspired by how human data scientists tackle challenges. It starts by generating a set of initial solution drafts and then iteratively refines and improves them based on performance feedback. This process is driven by a technique we call Solution Space Tree Search.

At its core, Solution Space Tree Search consists of three main components:

- **Solution Generator**: This component proposes new solutions by either creating novel drafts or making changes to existing solutions, such as fixing bugs or introducing improvements.
- **Evaluator**: The evaluator assesses the quality of each proposed solution by running it and comparing its performance against the objective. This is implemented by instructing the LLM to include statements that print the evaluation metric and by having another LLM parse the printed logs to extract the evaluation metric.
- **Base Solution Selector**: The solution selector picks the most promising solution from the explored options to serve as the starting point for the next iteration of refinement.

By repeatedly applying these steps, AIDE navigates the vast space of possible solutions, progressively refining its approach until it converges on the optimal solution for the given data science problem.

![Tree Search Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)





```bash

aide data_dir="aide/example_tasks/spooky-author-identification" \
      goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" \
      eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." \
      agent.code.model=deepseek-r1 \
      agent.steps=3 \
      agent.code.max_new_tokens=2048 \
      exec.timeout=600 \
      agent.code.temp=0.6 \
      inference_engine=vllm




.aide-ds/bin/python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda &

aide data_dir="aide/example_tasks/nomad2018-predict-transparent-conductors/dataset" \
     goal="Predict formation energy (formation_energy_ev_natom) and bandgap energy (bandgap_energy_ev) for materials given their composition and structural properties" \
     eval="Mean of column-wise Root Mean Squared Logarithmic Error (RMSLE) across the two target columns" \
     agent.code.model="deepseek-r1:latest" \
     agent.steps=20 \
     agent.code.max_new_tokens=2048 \
     agent.code.temp=0.6 \
     wandb.project="aide-nomad2018" \
     exp_name="32b_nomad2018"


```