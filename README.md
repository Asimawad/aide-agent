# AIDE Agent - Autonomous AI for Data Science

Open source autonomous AI agent powered by LLMs for data science and machine learning tasks.

## 🚀 Quick Start

### TPU Installation
```bash
# Option 1: One command (recommended)
uv pip install -e ".[tpu]"

# Option 2: Automated script
./install-tpu-fast.sh
```

### CUDA/GPU Installation
```bash
# Option 1: One command (recommended)
uv pip install -e ".[cuda]"

# Option 2: Automated script
./install-cuda-fast.sh
```

**That's it!** All dependencies including vLLM are installed in one go.

## 📦 What Gets Installed

### TPU (`[tpu]`)
- **vllm-tpu** - Complete TPU stack including:
  - PyTorch + torch-xla
  - JAX + jaxlib
  - tpu-inference backend (optimized)
  - libtpu runtime
- **150+ packages** - All data science, ML, and AI dependencies
- **AIDE-DS** - The autonomous agent framework

### CUDA (`[cuda]`)
- **vLLM** - High-performance LLM inference
- **PyTorch** - torch + torchvision + torchaudio (2.6.0)
- **bitsandbytes** - Quantization support
- **150+ packages** - All data science, ML, and AI dependencies
- **AIDE-DS** - The autonomous agent framework

## 🏃 Running AIDE

### Start vLLM Server

**TPU:**
```bash
vllm serve "Qwen/Qwen2.5-0.5B-Instruct" \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --trust-remote-code
```

**CUDA:**
```bash
vllm serve "Qwen/Qwen2.5-0.5B-Instruct" \
    --port 8000 \
    --device cuda \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --trust-remote-code
```

### Run the Agent
```bash
# Set your model
export CODER_MODEL="your-model-name"

# Run using entrypoint
./entrypoint.sh
```

## 📖 Documentation

- **[INSTALL.md](./INSTALL.md)** - Complete installation guide
- **[QUICKSTART.md](./QUICKSTART.md)** - Quick reference commands
- **[SETUP_SUMMARY.md](./SETUP_SUMMARY.md)** - Architecture and design decisions
- **Configuration**: Edit `aide/utils/config.yaml` for agent settings

## 📊 Features

- Autonomous data science workflows
- Multi-step reasoning and planning
- Code generation and execution
- Automated model training and evaluation
- Experiment tracking with Weights & Biases
- Support for tabular, text, image, and audio data
- TPU and CUDA/GPU acceleration
- Multiple ML frameworks (scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow)

## 🔧 Configuration

The project uses modern Python packaging with `pyproject.toml`:

```toml
[project.optional-dependencies]
tpu = ["vllm-tpu>=0.11.1", "tensorflow-cpu>=2.15.0"]
cuda = ["vllm>=0.7.0", "torch==2.6.0", "torchvision", "torchaudio", "bitsandbytes"]
dev = ["pytest", "black", "ruff", "mypy"]
```

Install with extras:
```bash
uv pip install -e ".[tpu-all]"    # TPU + dev tools
uv pip install -e ".[cuda-all]"   # CUDA + dev tools
```

## 🛠️ Development

```bash
# Install with development dependencies
uv pip install -e ".[tpu-all]"   # or .[cuda-all]

# Run tests
pytest

# Format code
black .

# Lint code
ruff check .
```

## 📝 Requirements

- **Python**: 3.11+
- **Environment**: TPU VM (v3/v4/v5) or CUDA GPU
- **RAM**: 32GB+ recommended
- **Disk**: 50GB+ free space
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (fast Python package installer)

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🎯 Monitoring

### TPU
```bash
# Check TPU info
tpu-info

# Monitor memory utilization
tpu-info --utilization

# Watch in real-time
watch -n 1 tpu-info --utilization
```

### CUDA
```bash
# Check GPU status
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

## 🐛 Troubleshooting

### Fresh Installation
```bash
# Remove old environment
rm -rf .venv

# Create new environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install
uv pip install -e ".[tpu]"   # or .[cuda]
```

### TPU: Check Devices
```bash
ls /dev/accel*
tpu-info
```

### CUDA: Check GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Verify vLLM
```bash
vllm --version
python -c "import vllm; print(vllm.__version__)"
```

## 🏗️ Project Structure

```
aide-agent/
├── pyproject.toml              # All dependencies unified here
├── aide/                       # Main agent code
│   ├── backend/                # Backend logic
│   ├── utils/                  # Utilities and config
│   └── run.py                  # Entry point
├── entrypoint.sh               # vLLM server startup
├── install-tpu-fast.sh         # Automated TPU setup
├── install-cuda-fast.sh        # Automated CUDA setup
├── INSTALL.md                  # Installation guide
└── QUICKSTART.md               # Quick reference
```

## 🌟 Key Features

### Multi-Framework Support
- **Traditional ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **NLP**: transformers, nltk, gensim, spacy
- **Computer Vision**: opencv, albumentations, timm
- **Audio**: librosa

### Data Science Tools
- pandas, numpy, scipy
- matplotlib, seaborn
- Bayesian optimization, Optuna
- Experiment tracking with W&B

### LLM Serving
- High-performance inference with vLLM
- TPU and CUDA acceleration
- Tensor parallelism support
- Multiple model formats (HuggingFace, GGUF, etc.)

## 📜 License

MIT License - see [LICENSE](./LICENSE) file

## 🙏 Acknowledgments

- Built with [vLLM](https://github.com/vllm-project/vllm) for LLM inference
- TPU support via [vllm-tpu](https://github.com/vllm-project/tpu-inference)
- Powered by [JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/), and [transformers](https://huggingface.co/docs/transformers)
- Fast package management with [uv](https://github.com/astral-sh/uv)

## 🔗 Links

- **Repository**: https://github.com/Asimawad/aide-agent
- **Issues**: https://github.com/Asimawad/aide-agent/issues
- **vLLM Documentation**: https://docs.vllm.ai/
- **vLLM TPU Backend**: https://github.com/vllm-project/tpu-inference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Made with ❤️ for the AI and Data Science community**

```

**you absolutely need a requirements.txt for legacy tools, generate it:**
pip install pip-tools
pip-compile pyproject.toml --extra=tpu -o requirements-tpu.txt
pip-compile pyproject.toml --extra=cuda -o requirements-cuda.txt
```