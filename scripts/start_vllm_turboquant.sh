#!/usr/bin/env bash

# === Configuration ===
PORT=8000
MODEL_ID="cyankiwi/Qwen3.5-35B-A3B-AWQ-8bit"
MODEL_NAME="qwen3.5-35b"
METADATA_PATH="$HOME/vllm-turboquant/calibration/turboquant_kv.json"
VENV_PATH="$HOME/.venvs/vllm-tq"

# === Cleanup Tripwire ===
cleanup() {
  echo ""
  echo "========================================================"
  echo "🛑 Shutdown signal received. Cleaning up..."
  
  if [ -n "${VLLM_PID:-}" ]; then
    echo "Stopping vLLM TurboQuant engine (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    # Give it a moment to shut down gracefully
    sleep 2
    kill -9 $VLLM_PID 2>/dev/null || true
  fi
  
  echo "✅ Server safely shut down. Goodbye!"
  echo "========================================================"
  exit 0
}
# Catch standard kill signals and terminal closes
trap cleanup INT TERM HUP QUIT EXIT

# === Activate Environment ===
source "$VENV_PATH/bin/activate"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_HOME=/usr/local/cuda
export HF_HUB_OFFLINE=1

# === Start vLLM TurboQuant Engine ===
echo "========================================================"
echo "🚀 Starting DGX Inference Engine (vLLM + TurboQuant)"
echo "🧠 Model: Qwen 3.5 35B-A3B (AWQ 8-bit)"
echo "⚡ KV Cache: TurboQuant35 (3.5-bit — 4.5× memory savings)"
echo "📏 Context Window: 262,144 tokens (256K)"
echo "🎯 Attention Backend: Triton"
echo "========================================================"

vllm serve "$MODEL_ID" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.70 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant \
  --turboquant-metadata-path "$METADATA_PATH" \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --host 0.0.0.0 \
  --port $PORT &

VLLM_PID=$!

# Wait for the server to be ready
echo ""
echo "⏳ Waiting for engine to initialize..."
for i in $(seq 1 120); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    break
  fi
  sleep 2
done

if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
  echo ""
  echo "========================================================"
  echo "✅ DGX TURBOQUANT ENGINE ONLINE AND BROADCASTING!"
  echo "🌐 Connect from your Mac: http://$(hostname -I | awk '{print $1}'):${PORT}"
  echo "🔗 OpenAI API: http://$(hostname -I | awk '{print $1}'):${PORT}/v1"
  echo "🧠 Engine PID: $VLLM_PID"
  echo "📊 Model: $MODEL_NAME"
  echo "⚡ TurboQuant35 KV cache active"
  echo "💡 To safely shut down, press Ctrl+C"
  echo "========================================================"
else
  echo ""
  echo "⚠️  Server may still be loading. Check logs above."
  echo "🔗 Endpoint will be at: http://$(hostname -I | awk '{print $1}'):${PORT}/v1"
fi

# The script pauses here and keeps the server alive
wait $VLLM_PID
