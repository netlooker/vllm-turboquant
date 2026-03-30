#!/usr/bin/env python3
"""Generate approximate TurboQuant metadata without full calibration.

This creates a "good-enough" metadata file using uniformly-spaced outlier
indices. It is NOT as accurate as the full activation-energy calibration,
but it lets you start serving immediately.

Usage:
    python3 generate_quick_metadata.py \
        --model cyankiwi/Qwen3.5-35B-A3B-AWQ-8bit \
        --kv-cache-dtype turboquant35 \
        --output calibration/turboquant_kv.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm.transformers_utils.config import get_config
from vllm.v1.attention.ops.turboquant_kv_cache import get_turboquant_outlier_count


def derive_model_shape(model: str, trust_remote_code: bool = False):
    config = get_config(model, trust_remote_code=trust_remote_code)
    text_config = getattr(config, "text_config", None)
    source = text_config if text_config is not None else config

    head_size = getattr(source, "head_dim", None)
    hidden_size = getattr(source, "hidden_size", None)
    num_attention_heads = getattr(source, "num_attention_heads", None)
    num_kv_heads = getattr(source, "num_key_value_heads", num_attention_heads)
    num_hidden_layers = getattr(source, "num_hidden_layers", None)

    if head_size is None and (hidden_size is not None and num_attention_heads is not None):
        head_size = hidden_size // num_attention_heads

    layer_types = getattr(source, "layer_types", None)
    if layer_types is not None:
        layer_types = tuple(layer_types)

    return head_size, num_kv_heads, num_hidden_layers, layer_types


def generate_uniform_indices(num_kv_heads: int, head_size: int, outlier_count: int):
    """Generate evenly-spaced outlier indices across the head dimension."""
    step = max(1, head_size // (outlier_count + 1))
    indices = []
    for _ in range(num_kv_heads):
        head_indices = sorted([(i * step) % head_size for i in range(1, outlier_count + 1)])
        # Deduplicate and fill if needed
        head_indices = sorted(set(head_indices))
        while len(head_indices) < outlier_count:
            for candidate in range(head_size):
                if candidate not in head_indices:
                    head_indices.append(candidate)
                    head_indices.sort()
                    if len(head_indices) >= outlier_count:
                        break
        indices.append(head_indices[:outlier_count])
    return indices


def main():
    parser = argparse.ArgumentParser(description="Generate quick TurboQuant metadata")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--kv-cache-dtype", choices=("turboquant25", "turboquant35"),
                        required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--layer-pattern", default="model.layers.{i}.self_attn.attn")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print(f"Reading model config from: {args.model}")
    head_size, num_kv_heads, num_hidden_layers, layer_types = derive_model_shape(
        args.model, trust_remote_code=args.trust_remote_code
    )
    print(f"  head_size={head_size}, num_kv_heads={num_kv_heads}, "
          f"num_layers={num_hidden_layers}")

    outlier_count = get_turboquant_outlier_count(head_size, args.kv_cache_dtype)
    print(f"  outlier_count={outlier_count} for {args.kv_cache_dtype}")

    # Determine which layers have attention
    if layer_types is not None:
        attn_layers = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    else:
        attn_layers = list(range(num_hidden_layers))

    # Build metadata
    layers = {}
    for layer_idx in attn_layers:
        layer_name = args.layer_pattern.format(i=layer_idx)
        indices = generate_uniform_indices(num_kv_heads, head_size, outlier_count)
        layers[layer_name] = {
            "key_high_precision_indices": indices,
            "value_high_precision_indices": indices,
        }

    metadata = {
        "version": 1,
        "recipe": args.kv_cache_dtype,
        "head_size": head_size,
        "model_name": args.model,
        "layers": layers,
        "calibration": {
            "method": "uniform_spacing_v1",
            "objective": "evenly_distributed",
            "num_prompts": 0,
            "max_seq_len": 0,
            "batch_size": 0,
            "num_observed_tokens": 0,
            "dtype": "none",
            "device": "none",
            "prompts_sha256": "none",
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written to: {output}")
    print(f"  {len(layers)} layers, {outlier_count} outliers per head")
    print(f"\n⚠  This is approximate metadata. For best quality, run full calibration later:")
    print(f"   python3 generate_turboquant_metadata.py --model {args.model} "
          f"--calibration-model <base-model> --kv-cache-dtype {args.kv_cache_dtype} ...")


if __name__ == "__main__":
    main()
