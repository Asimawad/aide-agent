# TPU Setup Guide for AIDE Agent

This guide explains how to install and run AIDE Agent with vLLM on Google Cloud TPUs.

## Prerequisites

- A Google Cloud TPU VM (v4, v5e, v5p, or v6e)
- Ubuntu 22.04 (recommended)
- Sufficient permissions to install system packages

## Quick Start (Automated)

The easiest way to set up TPU support is to use the provided setup script:

```bash
# 1. SSH into your TPU VM
gcloud compute tpus tpu-vm ssh YOUR-TPU-NAME --zone YOUR-ZONE

# 2. Clone this repository
git clone https://github.com/Asimawad/aide-agent.git
cd aide-agent

# 3. Run the automated setup script
./setup_tpu.sh
```

This script will:
- Install Miniconda
- Create a Python 3.11 environment
- Install PyTorch with TPU support (torch-xla)
- Clone and build vLLM with TPU support
- Install all AIDE dependencies

**Time required:** 15-20 minutes

## Manual Installation (Advanced)

If you prefer to install manually or need more control:

### Step 1: Install PyTorch with TPU Support

```bash
pip install torch~=2.6.0
pip install torch-xla[tpu]~=2.6.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### Step 2: Clone vLLM Repository

```bash
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### Step 3: Install vLLM TPU Requirements

The vLLM repo has a TPU-specific requirements file. The location may vary by version:

```bash
# Try this first (newer versions):
pip install -r requirements-tpu.txt

# If that doesn't exist, try:
pip install -r requirements/tpu.txt
```

### Step 4: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends --yes \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev
```

### Step 5: Build vLLM with TPU Support

This is the key step - you MUST set the environment variable before installation:

```bash
# This tells vLLM to compile for TPU instead of CUDA
export VLLM_TARGET_DEVICE="tpu"

# Install vLLM in development mode
pip install -e .
```

**Note:** This build process can take 10-15 minutes.

### Step 6: Install AIDE Agent

```bash
cd /path/to/aide-agent

# Install AIDE dependencies (excluding PyTorch/vLLM which are already installed)
pip install -r requirements-tpu.txt

# Install AIDE in development mode
pip install -e .
```

## Verification

After installation, verify everything is working:

### 1. Check TPU Access

```bash
python -c "import torch_xla.core.xla_model as xm; print(f'TPU device: {xm.xla_device()}')"
```

Expected output: `TPU device: xla:0` (or similar)

### 2. Check vLLM Installation

```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### 3. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --port 8000 \
    --dtype bfloat16 \
    --device tpu \
    --tensor-parallel-size 4 \
    --trust-remote-code
```

### 4. Test Health Endpoint

In another terminal:

```bash
curl http://localhost:8000/health
```

Expected output: `{"status":"ok"}` or similar

### 5. Test Inference

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "prompt": "def add_numbers(a, b):",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Running AIDE with TPU

Once vLLM is running, you can start AIDE experiments:

```bash
# Make sure vLLM server is running first (in another terminal or background)

# Run AIDE experiment
aide \
    data_dir="aide/example_tasks/house_prices" \
    goal="Predict the sales price for each house" \
    eval="Use the RMSE metric between the logarithm of the predicted and observed values." \
    agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    agent.steps=25
```

## Understanding the TPU Installation

### Why can't I just `pip install vllm`?

Regular `pip install vllm` compiles vLLM for CUDA (NVIDIA GPUs) by default. TPUs require a different compilation target, which is why you need:

1. **Build from source**: Clone the vLLM repository
2. **Set the target device**: `VLLM_TARGET_DEVICE="tpu"`
3. **Install in editable mode**: `pip install -e .`

This compiles vLLM specifically for TPU hardware.

### What is `requirements-tpu.txt` (from vLLM repo)?

The vLLM repository contains a file called `requirements-tpu.txt` (or `requirements/tpu.txt`) that includes TPU-specific dependencies like:
- `torch-xla` - PyTorch TPU extension
- TPU runtime libraries
- Other TPU-specific packages

This is **different** from the `requirements-tpu.txt` in the AIDE repo, which contains AIDE's dependencies (excluding PyTorch/vLLM).

### Installation Order Matters!

```
1. PyTorch + torch-xla (TPU support)
2. vLLM built with VLLM_TARGET_DEVICE="tpu"
3. AIDE dependencies
4. AIDE itself
```

Installing in the wrong order can cause conflicts or incorrect builds.

## Troubleshooting

### Error: "No TPU devices found"

```bash
# Check if TPU is recognized
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"

# If empty, you may not be on a TPU VM or torch-xla isn't installed correctly
```

### Error: "vLLM was built for CUDA, not TPU"

This means vLLM was installed without `VLLM_TARGET_DEVICE="tpu"`. Reinstall:

```bash
cd ~/vllm
pip uninstall vllm -y
VLLM_TARGET_DEVICE="tpu" pip install -e .
```

### Error: "requirements-tpu.txt not found" in vLLM repo

Try `requirements/tpu.txt` instead - the file location changed in different vLLM versions.

### Build takes too long or fails

- Make sure you have enough disk space (>20GB free)
- Check you have sufficient memory (>16GB recommended)
- TPU VMs sometimes have limited CPU - the build may take 15-20 minutes

## Additional Resources

- [vLLM TPU Documentation](https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [PyTorch XLA Documentation](https://pytorch.org/xla/release/2.6/index.html)

## Support

If you encounter issues:

1. Check the logs in `./logs/vllm_coder.log`
2. Verify TPU access with the commands above
3. Ensure you're using compatible versions (PyTorch 2.6.x, torch-xla 2.6.x)

