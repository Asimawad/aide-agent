#!/bin/bash
set -e
set -o pipefail

echo "=========================================="
echo "AIDE Agent TPU Setup Script"
echo "=========================================="

# Check if running on TPU VM
if ! command -v lspci &> /dev/null || ! lspci | grep -qi "google"; then
    echo "WARNING: This doesn't appear to be a Google Cloud TPU VM"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/bin/activate
    rm miniconda.sh
else
    echo "Conda already installed"
    source $(conda info --base)/etc/profile.d/conda.sh
fi

# 2. Create and activate conda environment
ENV_NAME="aide-tpu"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists, removing it..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating conda environment: $ENV_NAME with Python 3.11..."
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

# 3. Install PyTorch and torch-xla for TPU
echo "Installing PyTorch with TPU support..."
pip install --upgrade pip
pip install torch~=2.6.0
pip install torch-xla[tpu]~=2.6.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# 4. Clone vLLM repository
VLLM_DIR="$HOME/vllm"
if [ -d "$VLLM_DIR" ]; then
    echo "vLLM directory exists, updating..."
    cd "$VLLM_DIR"
    git pull
else
    echo "Cloning vLLM repository..."
    git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    cd "$VLLM_DIR"
fi

# 5. Install TPU-specific requirements from vLLM
echo "Installing vLLM TPU requirements..."
if [ -f "requirements-tpu.txt" ]; then
    pip install -r requirements-tpu.txt
else
    echo "Warning: requirements-tpu.txt not found, trying requirements/tpu.txt..."
    if [ -f "requirements/tpu.txt" ]; then
        pip install -r requirements/tpu.txt
    else
        echo "ERROR: Could not find TPU requirements file"
        exit 1
    fi
fi

# 6. Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install --no-install-recommends --yes \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    git \
    curl

# 7. Build and install vLLM with TPU support
echo "Building vLLM with TPU support (this may take 10-15 minutes)..."
export VLLM_TARGET_DEVICE="tpu"
pip install -e .

# 8. Verify vLLM installation
echo "Verifying vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# 9. Install aide-agent dependencies
echo "Installing AIDE agent..."
cd /home/asim/aide-agent

# Install only non-conflicting dependencies
echo "Installing AIDE dependencies (excluding PyTorch/vLLM)..."
pip install accelerate>=1.6.0
pip install backoff wandb>=0.19.10
pip install bitsandbytes>=0.45.5
pip install black coolname dataclasses-json funcy genson humanize
pip install ipython jsonschema omegaconf
pip install openai==1.65.0
pip install pandas psutil>=5.9.0 python-dotenv python-igraph
pip install rich shutup tqdm transformers
pip install setuptools packaging wheel s3fs
pip install tensorflow datasets scikit-learn xgboost lightgbm
pip install keras matplotlib albumentations seaborn inflect
pip install statsmodels nltk gensim peft pypdf pytest
pip install rouge-score pytorch-lightning sacrebleu
pip install scikit-image scikit-optimize scipy spacy
pip install torchmetrics torchtext torchinfo torch-geometric
pip install catboost timm opencv-python Pillow
pip install librosa bayesian-optimization optuna
pip install httpx igraph cairocffi radon
pip install tensorflow-decision-forests pymorphy2 num2words

# Install aide-agent in development mode
pip install -e .

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify TPU access:"
echo "  python -c 'import torch_xla.core.xla_model as xm; print(xm.xla_device())'"
echo ""
echo "To start vLLM server:"
echo "  ./entrypoint.sh"
echo ""

