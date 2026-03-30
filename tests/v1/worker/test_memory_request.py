# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import CacheConfig
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.utils import request_memory


def test_request_memory_error_suggests_fitting_budget():
    snapshot = MemorySnapshot(
        free_memory=899,
        total_memory=1000,
        device=torch.device("cuda:0"),
        auto_measure=False,
    )
    cache_config = CacheConfig(gpu_memory_utilization=0.9)

    with pytest.raises(ValueError) as exc_info:
        request_memory(snapshot, cache_config)

    msg = str(exc_info.value)
    assert "--gpu-memory-utilization 0.8990" in msg
    assert "--kv-cache-memory-bytes 899" in msg


def test_request_memory_allows_manual_kv_cache_budget():
    snapshot = MemorySnapshot(
        free_memory=899,
        total_memory=1000,
        device=torch.device("cuda:0"),
        auto_measure=False,
    )
    cache_config = CacheConfig(
        gpu_memory_utilization=0.9,
        kv_cache_memory_bytes=512,
    )

    assert request_memory(snapshot, cache_config) == 900
