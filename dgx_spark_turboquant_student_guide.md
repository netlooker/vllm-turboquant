# 🚀 Guide: Deploying TurboQuant on NVIDIA DGX Spark

Welcome, Students! This guide will show you how to set up **vLLM with TurboQuant** on an NVIDIA DGX Spark workstation. 

TurboQuant is a special optimization that compresses the "KV Cache" (the model's memory) to just 3.5 bits. This allows you to run huge models like **Qwen 3.5 (35B parameters)** with a massive **256K context window** on a single machine!

---

## 🛠 Prerequisites

*   **Hardware**: NVIDIA DGX Spark (Grace Blackwell / GB10).
*   **Operating System**: Ubuntu 22.04+ (aarch64).
*   **Drivers**: CUDA 13.0+ installed.

---

## 📥 Step 1: Clone the Project

First, download the source code from GitHub:

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/vllm-turboquant.git
cd vllm-turboquant
```

---

## 🐍 Step 2: Setup Python Environment

We use `uv` because it is much faster than standard `pip`.

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

2.  **Create a Virtual Environment**:
    ```bash
    uv venv --python 3.12 ~/.venvs/vllm-tq
    source ~/.venvs/vllm-tq/bin/activate
    ```

---

## 📦 Step 3: Install Specialized Dependencies

We need specific versions of PyTorch and build tools for the Blackwell architecture.

1.  **Install PyTorch (CUDA 13)**:
    ```bash
    uv pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu130
    ```

2.  **Install Build Tools**:
    ```bash
    uv pip install "setuptools>=78" setuptools-scm wheel cmake ninja packaging
    ```

---

## 🏗 Step 4: Build vLLM from Source

Now we build the custom engine. We use the "precompiled" flag to speed things up significantly.

```bash
# Set environment variables for the build
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0"
export VLLM_TARGET_DEVICE=cuda
export VLLM_VERSION_OVERRIDE="0.18.1+turboquant"

# Install vLLM in editable mode
VLLM_USE_PRECOMPILED=1 uv pip install --no-build-isolation -e .
```

---

## 📊 Step 5: Prepare the Model & Metadata

We need to generate a "Map" (metadata) that tells TurboQuant how to compress the information.

1.  **Identify your model**:
    ```bash
    MODEL_ID="cyankiwi/Qwen3.5-35B-A3B-AWQ-8bit"
    ```

2.  **Generate Quick Metadata**:
    ```bash
    python3 calibration/generate_quick_metadata.py \
      --model "$MODEL_ID" \
      --kv-cache-dtype turboquant35 \
      --output calibration/turboquant_kv.json
    ```

---

## 🚀 Step 6: Launch the Engine!

You can now start the server using the provided helper script:

```bash
bash scripts/start_vllm_turboquant.sh
```

---

## 🧪 Step 7: Test Your AI

Open a new terminal and send a request to your new engine:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role":"user","content":"Hello DGX Spark! Who are you?"}],
    "max_tokens": 100
  }'
```

---

## 💡 Troubleshooting Tips

*   **Memory Errors**: If you see "Out of Memory", lower the `--gpu-memory-utilization` in the launch script.
*   **Version Mismatch**: Always ensure you have run `source ~/.venvs/vllm-tq/bin/activate` before starting.
*   **DNS Issues**: If the model won't download, check your `/etc/resolv.conf`.

---

**Congratulations!** You are now running state-of-the-art KV cache compression on the world's most advanced AI hardware.
