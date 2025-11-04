#!/usr/bin/env python3
"""
Small Model DDP Test (Issue #30, Phase 1.2)

Tests multi-GPU DDP initialization with a tiny model to isolate NCCL communication
issues from model-size-specific problems.

If this succeeds: Issue is likely related to large model size
If this fails: Issue is NCCL/cluster configuration

Usage:
    # As pytest (single GPU test):
    pytest tests/diagnostics/test_small_model_ddp.py -v -s

    # As standalone DDP script:
    torchrun --nproc_per_node=8 tests/diagnostics/test_small_model_ddp.py
    # Or test with fewer GPUs:
    torchrun --nproc_per_node=4 tests/diagnostics/test_small_model_ddp.py

Expected output:
    - All ranks initialize successfully
    - DDP wrapping succeeds
    - Forward pass completes
    - Gradient synchronization works
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyModel(nn.Module):
    """Tiny model for testing DDP communication."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.integration
def test_tiny_model_single_gpu():
    """Test TinyModel on single GPU (pytest mode)."""
    print("=" * 80)
    print("Testing TinyModel on Single GPU")
    print("=" * 80)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Create model
    model = TinyModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created: {param_count:,} parameters")

    # Test forward pass
    x = torch.randn(4, 100).to(device)
    y = model(x)
    print(f"Forward pass successful: output shape {y.shape}")

    # Test backward pass
    loss = y.sum()
    loss.backward()
    print(f"Backward pass successful")

    assert y.shape == (4, 10), f"Expected output shape (4, 10), got {y.shape}"
    print("✓ Single GPU test passed")


def run_ddp_initialization():
    """Run DDP initialization with tiny model across multiple GPUs.

    NOTE: This function is NOT a pytest test (doesn't start with test_).
    It should only be called by torchrun or from main() when running
    as a standalone script with torchrun.

    Usage:
        torchrun --nproc_per_node=8 tests/diagnostics/test_small_model_ddp.py
    """
    # Get rank info from environment (set by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}/{world_size}] Starting DDP initialization test")
    print(f"[Rank {rank}] Device: {device}")

    # Initialize process group
    print(f"[Rank {rank}] Initializing process group...")
    try:
        dist.init_process_group(backend="nccl")
        print(f"[Rank {rank}] ✓ Process group initialized")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Process group initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Create model and move to device
    print(f"[Rank {rank}] Creating model...")
    try:
        model = TinyModel().to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[Rank {rank}] ✓ Model created: {param_count:,} parameters")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Wrap with DDP
    print(f"[Rank {rank}] Wrapping model with DDP...")
    try:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] ✓ DDP wrapping successful")
    except Exception as e:
        print(f"[Rank {rank}] ✗ DDP wrapping failed: {e}")
        import traceback

        traceback.print_exc()
        dist.destroy_process_group()
        return False

    # Test forward pass
    print(f"[Rank {rank}] Testing forward pass...")
    try:
        x = torch.randn(4, 100).to(device)
        y = model(x)
        print(f"[Rank {rank}] ✓ Forward pass successful: output shape {y.shape}")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        dist.destroy_process_group()
        return False

    # Test backward pass (gradient sync)
    print(f"[Rank {rank}] Testing backward pass...")
    try:
        loss = y.sum()
        loss.backward()
        print(f"[Rank {rank}] ✓ Backward pass successful")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Backward pass failed: {e}")
        import traceback

        traceback.print_exc()
        dist.destroy_process_group()
        return False

    # Synchronize all ranks
    print(f"[Rank {rank}] Synchronizing all ranks...")
    try:
        dist.barrier()
        print(f"[Rank {rank}] ✓ Barrier synchronization successful")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Barrier synchronization failed: {e}")
        import traceback

        traceback.print_exc()
        dist.destroy_process_group()
        return False

    # Cleanup
    print(f"[Rank {rank}] Cleaning up...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] ✓ Process group destroyed")

    return True


def main():
    """Run small model DDP test."""

    # Check environment variables
    required_vars = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]
    missing_vars = [v for v in required_vars if v not in os.environ]

    if missing_vars:
        print("ERROR: This script must be run with torchrun")
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("\nUsage:")
        print("  torchrun --nproc_per_node=8 tests/diagnostics/test_small_model_ddp.py")
        return 1

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # Only print header from rank 0
    if rank == 0:
        print("=" * 80)
        print("Small Model DDP Test")
        print("Issue #30, Phase 1.2")
        print("=" * 80)
        print(f"World size: {world_size} GPUs")
        print("=" * 80)

    # Run test
    success = run_ddp_initialization()

    # Barrier to ensure all ranks finish
    if "RANK" in os.environ:
        try:
            if dist.is_initialized():
                dist.barrier()
        except:
            pass

    # Print results
    if rank == 0:
        print("\n" + "=" * 80)
        if success:
            print("✓ Small Model DDP Test PASSED")
            print(f"Multi-GPU DDP works with {world_size} GPUs")
            print("Issue likely specific to large model size")
        else:
            print("✗ Small Model DDP Test FAILED")
            print("Issue is with NCCL/cluster configuration, not model size")
        print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
