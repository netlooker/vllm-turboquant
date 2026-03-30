#!/usr/bin/env bash
# =============================================================================
# vllm-turboquant — Full Deployment on NVIDIA DGX Spark
#
# This script automates the entire installation process:
#   1. NGC container login (if Docker path)
#   2. Clone the repo
#   3. Build from source OR Docker overlay
#   4. Generate TurboQuant calibration metadata
#   5. Launch the server
#
# Usage:
#   # Docker path (recommended):
#   bash deploy_dgx_spark.sh --docker
#
#   # Source build path:
#   bash deploy_dgx_spark.sh --source
#
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

INSTALL_MODE="${1:---source}"
REPO_URL="https://github.com/mitkox/vllm-turboquant.git"
WORK_DIR="${HOME}/vllm-turboquant"
VENV_DIR="${HOME}/.venvs/vllm-turboquant"
MODEL="${MODEL:-cyankiwi/Qwen3.5-27B-AWQ-4bit}"
# For calibration, use the non-quantized base model
CALIBRATION_MODEL="${CALIBRATION_MODEL:-Qwen/Qwen2.5-27B}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant35}"
METADATA_PATH="${WORK_DIR}/calibration/turboquant_kv.json"
NGC_IMAGE="nvcr.io/nvidia/vllm:26.02-py3"

echo -e "\n${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  vllm-turboquant — DGX Spark Deployment              ║${NC}"
echo -e "${CYAN}║  Mode: ${INSTALL_MODE}                                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"

# ─────────────────────────────────────────────────────────────
# Step 0: Pre-flight checks
# ─────────────────────────────────────────────────────────────
log "Step 0: Pre-flight checks..."

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  err "This script is designed for DGX Spark (aarch64). Detected: $ARCH"
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  err "nvidia-smi not found. Ensure NVIDIA drivers are installed."
  exit 1
fi

if ! command -v git &>/dev/null; then
  log "Installing git..."
  sudo apt-get update && sudo apt-get install -y git
fi

# ─────────────────────────────────────────────────────────────
# Step 1: Clone the repository
# ─────────────────────────────────────────────────────────────
log "Step 1: Cloning vllm-turboquant..."

if [ -d "$WORK_DIR" ]; then
  log "Repository already exists at $WORK_DIR, pulling latest..."
  cd "$WORK_DIR" && git pull
else
  git clone "$REPO_URL" "$WORK_DIR"
  cd "$WORK_DIR"
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Set environment variables
# ─────────────────────────────────────────────────────────────
log "Step 2: Setting environment variables..."

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="12.0"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=4
export NVCC_THREADS=2
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# ═════════════════════════════════════════════════════════════
# DOCKER PATH
# ═════════════════════════════════════════════════════════════
if [ "$INSTALL_MODE" = "--docker" ]; then

  # ───────────────────────────────────────────────────────────
  # Step 3D: NGC Container Registry Setup
  # ───────────────────────────────────────────────────────────
  log "Step 3: Setting up NGC Container Registry..."

  if ! command -v docker &>/dev/null; then
    err "Docker is not installed. Install it first:"
    echo "  sudo apt-get update && sudo apt-get install -y docker.io"
    echo "  sudo systemctl enable --now docker"
    echo "  sudo usermod -aG docker \$USER"
    echo "  # Log out and back in, then re-run this script"
    exit 1
  fi

  # Check if already logged in to NGC
  if ! docker pull "$NGC_IMAGE" --quiet 2>/dev/null; then
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  NGC Container Registry Setup${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────${NC}"
    echo ""
    echo "To pull NVIDIA containers, you need a free NGC account:"
    echo ""
    echo "  1. Go to: https://ngc.nvidia.com/signin"
    echo "  2. Create a free account (or sign in with Google/NVIDIA)"
    echo "  3. Go to: https://ngc.nvidia.com/setup/api-key"
    echo "  4. Click 'Generate API Key'"
    echo "  5. Copy the key"
    echo ""
    echo "Now log in to the container registry:"
    echo ""
    echo -e "  ${GREEN}docker login nvcr.io${NC}"
    echo "  Username: \$oauthtoken"
    echo "  Password: <paste your NGC API key>"
    echo ""
    read -rp "Press Enter after you've logged in to NGC... "
    docker pull "$NGC_IMAGE"
  fi

  log "NGC image pulled successfully."

  # ───────────────────────────────────────────────────────────
  # Step 4D: Build Docker overlay
  # ───────────────────────────────────────────────────────────
  log "Step 4: Building Docker overlay..."

  cat > "${WORK_DIR}/Dockerfile.turboquant" << 'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
FROM ${BASE_IMAGE}

# TurboQuant core ops
COPY vllm/v1/attention/ops/turboquant_kv_cache.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/
COPY vllm/v1/attention/ops/triton_turboquant_decode.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/
COPY vllm/v1/attention/ops/triton_turboquant_kv_update.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/
COPY vllm/v1/attention/ops/turboquant_metadata.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/

# Modified integration files
COPY vllm/config/cache.py \
     /usr/local/lib/python3.12/dist-packages/vllm/config/
COPY vllm/engine/arg_utils.py \
     /usr/local/lib/python3.12/dist-packages/vllm/engine/
COPY vllm/v1/attention/selector.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/
COPY vllm/v1/attention/backends/triton_attn.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/
COPY vllm/v1/kv_cache_interface.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/
COPY vllm/platforms/cuda.py \
     /usr/local/lib/python3.12/dist-packages/vllm/platforms/
COPY vllm/utils/torch_utils.py \
     /usr/local/lib/python3.12/dist-packages/vllm/utils/
COPY vllm/model_executor/layers/attention/attention.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/

# Tools & calibration data
COPY benchmarks/generate_turboquant_metadata.py /workspace/
COPY benchmarks/run_turboquant_gb10_compare.sh /workspace/
COPY calibration/ /workspace/calibration/

WORKDIR /workspace
DOCKERFILE

  cd "$WORK_DIR"
  docker build \
    --build-arg BASE_IMAGE="$NGC_IMAGE" \
    -f Dockerfile.turboquant \
    -t vllm-turboquant:latest .

  log "Docker image built: vllm-turboquant:latest"

  # ───────────────────────────────────────────────────────────
  # Step 5D: Generate calibration metadata (in container)
  # ───────────────────────────────────────────────────────────
  log "Step 5: Generating TurboQuant calibration metadata..."
  log "(This will download the calibration model and run inference — may take 15-30 min)"

  mkdir -p "${WORK_DIR}/calibration"

  docker run --rm --gpus all \
    -v "${WORK_DIR}/calibration:/workspace/calibration" \
    -e PYTORCH_ALLOC_CONF="expandable_segments:True" \
    vllm-turboquant:latest \
    python3 /workspace/generate_turboquant_metadata.py \
      --model "${MODEL}" \
      --calibration-model "${CALIBRATION_MODEL}" \
      --kv-cache-dtype "${KV_CACHE_DTYPE}" \
      --prompts-file /workspace/calibration/prompts.txt \
      --output /workspace/calibration/turboquant_kv.json \
      --dtype bfloat16 \
      --device auto \
      --max-prompts 100 \
      --batch-size 2 \
      --max-seq-len 2048

  log "Metadata generated at: ${WORK_DIR}/calibration/turboquant_kv.json"

  # ───────────────────────────────────────────────────────────
  # Step 6D: Launch the server
  # ───────────────────────────────────────────────────────────
  log "Step 6: Launching vLLM server with TurboQuant..."

  docker run -d --gpus all \
    --name vllm-turboquant-server \
    -v "${WORK_DIR}/calibration:/workspace/calibration" \
    -p 8000:8000 \
    -e LD_LIBRARY_PATH="/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}" \
    -e PYTORCH_ALLOC_CONF="expandable_segments:True" \
    vllm-turboquant:latest \
    vllm serve "${MODEL}" \
      --tensor-parallel-size 1 \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.70 \
      --attention-backend TRITON_ATTN \
      --kv-cache-dtype "${KV_CACHE_DTYPE}" \
      --enable-turboquant \
      --turboquant-metadata-path /workspace/calibration/turboquant_kv.json \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 64

  log "Server starting... check logs with:"
  echo "  docker logs -f vllm-turboquant-server"
  echo ""
  log "API endpoint: http://localhost:8000/v1/chat/completions"

# ═════════════════════════════════════════════════════════════
# SOURCE BUILD PATH
# ═════════════════════════════════════════════════════════════
elif [ "$INSTALL_MODE" = "--source" ]; then

  # ───────────────────────────────────────────────────────────
  # Step 3S: Install uv and create venv
  # ───────────────────────────────────────────────────────────
  log "Step 3: Setting up Python environment..."

  if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.12 "$VENV_DIR"
  fi
  source "${VENV_DIR}/bin/activate"

  # ───────────────────────────────────────────────────────────
  # Step 4S: Install PyTorch for CUDA 13
  # ───────────────────────────────────────────────────────────
  log "Step 4: Installing PyTorch with CUDA 13..."

  uv pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu130

  # Verify
  python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
else:
    print('WARNING: CUDA not available via PyTorch')
"

  # ───────────────────────────────────────────────────────────
  # Step 5S: Install vllm-turboquant from source
  # ───────────────────────────────────────────────────────────
  log "Step 5: Building vllm-turboquant from source..."
  log "(This may take 30-60 minutes on DGX Spark)"

  cd "$WORK_DIR"

  # Install build deps
  uv pip install -r requirements/build.txt

  # Try precompiled first, fall back to full build
  log "Attempting precompiled install first..."
  if VLLM_USE_PRECOMPILED=1 uv pip install -e . 2>/dev/null; then
    log "Precompiled install succeeded!"
  else
    warn "Precompiled install failed, doing full source build..."
    uv pip install -e .
  fi

  # Verify
  python3 -c "
import vllm
from vllm.v1.attention.ops.turboquant_kv_cache import is_turboquant_kv_cache
print(f'vLLM {vllm.__version__} installed successfully')
print(f'TurboQuant available: {is_turboquant_kv_cache(\"turboquant35\")}')
"

  # ───────────────────────────────────────────────────────────
  # Step 6S: Generate calibration metadata
  # ───────────────────────────────────────────────────────────
  log "Step 6: Generating TurboQuant calibration metadata..."
  log "(This will download the calibration model and run inference — may take 15-30 min)"

  python3 benchmarks/generate_turboquant_metadata.py \
    --model "${MODEL}" \
    --calibration-model "${CALIBRATION_MODEL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --prompts-file calibration/prompts.txt \
    --output "${METADATA_PATH}" \
    --dtype bfloat16 \
    --device auto \
    --max-prompts 100 \
    --batch-size 2 \
    --max-seq-len 2048

  log "Metadata generated at: ${METADATA_PATH}"

  # ───────────────────────────────────────────────────────────
  # Step 7S: Launch the server
  # ───────────────────────────────────────────────────────────
  log "Step 7: Launching vLLM server with TurboQuant..."

  vllm serve "${MODEL}" \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.70 \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-turboquant \
    --turboquant-metadata-path "${METADATA_PATH}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64

else
  err "Unknown mode: $INSTALL_MODE"
  echo "Usage: $0 [--docker|--source]"
  exit 1
fi
