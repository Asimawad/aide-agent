#!/usr/bin/env bash
set -euo pipefail

echo "⏹ Stopping vLLM servers and any remaining Python processes…"

# 1. Kill any vLLM API servers
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server")
  echo " → Killing vLLM API server PIDs: $pids"
  kill $pids
else
  echo " → No vLLM API server processes found."
fi

# 2. Kill any stray Python in this venv (adjust the path if yours differs)
VENV_BIN="$(dirname "$(which python)")"
if pgrep -f "$VENV_BIN/python" >/dev/null; then
  pids=$(pgrep -f "$VENV_BIN/python")
  echo " → Killing other Python processes in venv: $pids"
  kill $pids
else
  echo " → No other Python processes in venv."
fi

# 3. Wait a moment for processes to exit
sleep 2

echo "✅ All vLLM and Python processes stopped."

#!/usr/bin/env bash
set -euo pipefail

echo "🧹 Resetting GPUs and system caches…"

# 1. Reset each GPU (NVIDIA data-center GPUs only)
if command -v nvidia-smi >/dev/null; then
  for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    echo " → Resetting GPU #$idx"
    sudo nvidia-smi --gpu-reset -i "$idx" || echo "   ⚠️ GPU reset not supported on this device."
  done
else
  echo " → nvidia-smi not found, skipping GPU reset."
fi

# 2. Drop pagecache, dentries and inodes
echo " → Dropping OS caches"
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo "✅ System reset complete."
