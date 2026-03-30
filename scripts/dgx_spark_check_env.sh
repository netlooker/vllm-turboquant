#!/usr/bin/env bash
# =============================================================================
# DGX Spark Environment Check
# Run this on your DGX Spark to verify readiness for vllm-turboquant
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }

echo -e "\n${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  DGX Spark — vllm-turboquant Environment Check${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}\n"

# --- Platform ---
echo -e "${CYAN}[1/7] Platform${NC}"
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  ok "Architecture: $ARCH (ARM — correct for DGX Spark)"
else
  fail "Architecture: $ARCH (expected aarch64)"
fi

OS=$(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')
echo -e "  ℹ  OS: ${OS:-unknown}"

KERNEL=$(uname -r)
echo -e "  ℹ  Kernel: $KERNEL"

# --- GPU ---
echo -e "\n${CYAN}[2/7] GPU${NC}"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
  CUDA_DRIVER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
  ok "GPU: $GPU_NAME"
  ok "Driver: $DRIVER"
  ok "GPU Memory: $GPU_MEM"
  ok "CUDA (driver): $CUDA_DRIVER"
else
  fail "nvidia-smi not found"
fi

# --- CUDA Toolkit ---
echo -e "\n${CYAN}[3/7] CUDA Toolkit${NC}"
if command -v nvcc &>/dev/null; then
  NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
  ok "nvcc: $NVCC_VERSION"
else
  fail "nvcc not found — install CUDA toolkit"
fi

if [ -d "/usr/local/cuda" ]; then
  ok "CUDA_HOME: /usr/local/cuda exists"
else
  warn "CUDA_HOME: /usr/local/cuda not found"
fi

SBSA_LIB="/usr/local/cuda/targets/sbsa-linux/lib"
if [ -d "$SBSA_LIB" ]; then
  ok "SBSA libs: $SBSA_LIB exists"
else
  warn "SBSA libs: $SBSA_LIB not found (may need LD_LIBRARY_PATH adjustment)"
fi

# --- Python ---
echo -e "\n${CYAN}[4/7] Python${NC}"
if command -v python3 &>/dev/null; then
  PY_VERSION=$(python3 --version 2>&1)
  ok "$PY_VERSION"
else
  fail "python3 not found"
fi

if command -v pip3 &>/dev/null || python3 -m pip --version &>/dev/null 2>&1; then
  ok "pip available"
else
  warn "pip not found"
fi

# --- Docker ---
echo -e "\n${CYAN}[5/7] Docker & Container Runtime${NC}"
if command -v docker &>/dev/null; then
  DOCKER_V=$(docker --version 2>/dev/null)
  ok "Docker: $DOCKER_V"
else
  warn "Docker not installed"
fi

if command -v nvidia-container-cli &>/dev/null || dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
  ok "NVIDIA Container Toolkit: installed"
else
  warn "NVIDIA Container Toolkit: not detected"
fi

# --- Memory ---
echo -e "\n${CYAN}[6/7] Memory${NC}"
TOTAL_MEM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
AVAIL_MEM=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
echo -e "  ℹ  Total RAM: ${TOTAL_MEM:-?} GB (unified CPU+GPU)"
echo -e "  ℹ  Available: ${AVAIL_MEM:-?} GB"

SWAP=$(free -g 2>/dev/null | awk '/^Swap:/{print $2}')
echo -e "  ℹ  Swap: ${SWAP:-0} GB"

# --- Disk ---
echo -e "\n${CYAN}[7/7] Disk Space${NC}"
DISK_AVAIL=$(df -BG / 2>/dev/null | awk 'NR==2{print $4}')
echo -e "  ℹ  Root partition available: ${DISK_AVAIL:-?}"
if [ -d "/home" ]; then
  HOME_AVAIL=$(df -BG /home 2>/dev/null | awk 'NR==2{print $4}')
  echo -e "  ℹ  /home available: ${HOME_AVAIL:-?}"
fi

# --- Summary ---
echo -e "\n${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Copy-paste the output above and send it back so we can"
echo "determine the exact installation steps for your system."
echo ""
