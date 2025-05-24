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
