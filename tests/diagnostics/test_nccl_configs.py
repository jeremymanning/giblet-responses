#!/usr/bin/env python3
"""
NCCL Configuration Test (Issue #30, Phase 2.1)

Tests different NCCL environment configurations to find one that works.
Based on Phase 1.2 findings: shared memory communication is failing.

This script tries multiple NCCL configurations in sequence to identify
which settings allow DDP to work on tensor01/tensor02.

Usage:
    pytest tests/diagnostics/test_nccl_configs.py -v
    # Or run as standalone script:
    python tests/diagnostics/test_nccl_configs.py

Note: Run this BEFORE using torchrun, as it will spawn torchrun processes.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

# NCCL configurations to test (in priority order)
NCCL_CONFIGS = [
    {
        "name": "Force Socket Transport (Disable SHM)",
        "desc": "Disable shared memory, use sockets for all communication",
        "env": {
            "NCCL_SHM_DISABLE": "1",  # Disable shared memory
            "NCCL_P2P_DISABLE": "0",  # Keep P2P enabled
            "NCCL_DEBUG": "INFO",  # Enable debug logging
        },
    },
    {
        "name": "Disable P2P and SHM",
        "desc": "Use only socket transport, no P2P or shared memory",
        "env": {
            "NCCL_SHM_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",  # Disable P2P as well
            "NCCL_DEBUG": "INFO",
        },
    },
    {
        "name": "Increase Timeouts and Buffers",
        "desc": "Keep defaults but increase timeouts and buffer sizes",
        "env": {
            "NCCL_DEBUG": "INFO",
            "NCCL_TIMEOUT": "600",  # 10 minutes (default: 30s)
            "NCCL_BUFFSIZE": "33554432",  # 32MB (default: 4MB)
            "NCCL_NTHREADS": "4",  # More NCCL threads
        },
    },
    {
        "name": "InfiniBand with Socket Fallback",
        "desc": "Try InfiniBand but allow fallback to sockets",
        "env": {
            "NCCL_IB_DISABLE": "0",  # Try InfiniBand
            "NCCL_SOCKET_IFNAME": "eth0,ib0",  # Allow fallback
            "NCCL_DEBUG": "INFO",
            "NCCL_SHM_DISABLE": "1",  # Still disable SHM
        },
    },
    {
        "name": "Single Node Optimized",
        "desc": "Optimized for single-node multi-GPU (8 GPUs on one machine)",
        "env": {
            "NCCL_DEBUG": "INFO",
            "NCCL_SINGLE_RING_THRESHOLD": "1",  # Use single ring for small sizes
            "NCCL_SHM_DISABLE": "1",
            "NCCL_ALGO": "Ring",  # Force ring algorithm
        },
    },
]


def run_ddp_test(config, gpus=8):
    """
    Run small model DDP test with given NCCL configuration.

    Parameters
    ----------
    config : dict
        NCCL configuration dict with 'name', 'desc', and 'env' keys
    gpus : int
        Number of GPUs to use

    Returns
    -------
    success : bool
        True if test passed, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Testing Configuration: {config['name']}")
    print("=" * 80)
    print(f"Description: {config['desc']}")
    print("\nEnvironment variables:")
    for key, value in config["env"].items():
        print(f"  {key}={value}")
    print("=" * 80)

    # Prepare environment
    env = os.environ.copy()
    env.update(config["env"])

    # Run torchrun with small model test
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "tests/diagnostics/test_small_model_ddp.py",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 80)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        # Check if successful
        if result.returncode == 0:
            print("\n✓ TEST PASSED!")
            print("-" * 80)
            print("STDOUT (last 50 lines):")
            stdout_lines = result.stdout.split("\n")[-50:]
            print("\n".join(stdout_lines))
            print("-" * 80)
            return True
        else:
            print("\n✗ TEST FAILED")
            print(f"Exit code: {result.returncode}")
            print("-" * 80)
            print("STDERR (last 30 lines):")
            stderr_lines = result.stderr.split("\n")[-30:]
            print("\n".join(stderr_lines))
            print("-" * 80)
            print("STDOUT (last 30 lines):")
            stdout_lines = result.stdout.split("\n")[-30:]
            print("\n".join(stdout_lines))
            print("-" * 80)
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ TEST TIMED OUT (120s)")
        return False
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_nccl_configurations():
    """Test all NCCL configurations as pytest."""
    print("=" * 80)
    print("NCCL Configuration Test Suite")
    print("Issue #30, Phase 2.1")
    print("=" * 80)
    print(f"\nTesting {len(NCCL_CONFIGS)} different NCCL configurations...")
    print("This will take several minutes.\n")

    results = {}

    for i, config in enumerate(NCCL_CONFIGS, 1):
        print(f"\n{'=' * 80}")
        print(f"Configuration {i}/{len(NCCL_CONFIGS)}")
        print(f"{'=' * 80}")

        success = run_ddp_test(config)
        results[config["name"]] = success

        if success:
            print(f"\n✓ FOUND WORKING CONFIGURATION!")
            print(f"Config name: {config['name']}")
            print(f"\nRecommended environment variables:")
            for key, value in config["env"].items():
                print(f"export {key}={value}")
            break  # Stop testing once we find a working config

        # Wait a bit between tests
        time.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for config_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{config_name:50s}: {status}")

    working_configs = [name for name, passed in results.items() if passed]

    print("\n" + "=" * 80)
    if working_configs:
        print(f"✓ Found {len(working_configs)} working configuration(s):")
        for config_name in working_configs:
            print(f"  - {config_name}")
        print("\nApply the working configuration to scripts/train.py")
    else:
        print("✗ No working NCCL configurations found")
        print("\nNext steps:")
        print("1. Check /dev/shm permissions and size: df -h /dev/shm")
        print("2. Check NCCL installation: ldconfig -p | grep nccl")
        print("3. Try using FSDP instead of DDP (Strategy 2.4)")
        pytest.fail("No working NCCL configurations found")

    assert len(working_configs) > 0, "Expected at least one working configuration"


def main():
    """Test all NCCL configurations (standalone script mode)."""
    print("=" * 80)
    print("NCCL Configuration Test Suite")
    print("Issue #30, Phase 2.1")
    print("=" * 80)
    print(f"\nTesting {len(NCCL_CONFIGS)} different NCCL configurations...")
    print("This will take several minutes.\n")

    results = {}

    for i, config in enumerate(NCCL_CONFIGS, 1):
        print(f"\n{'=' * 80}")
        print(f"Configuration {i}/{len(NCCL_CONFIGS)}")
        print(f"{'=' * 80}")

        success = run_ddp_test(config)
        results[config["name"]] = success

        if success:
            print(f"\n✓ FOUND WORKING CONFIGURATION!")
            print(f"Config name: {config['name']}")
            print(f"\nRecommended environment variables:")
            for key, value in config["env"].items():
                print(f"export {key}={value}")
            break  # Stop testing once we find a working config

        # Wait a bit between tests
        time.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for config_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{config_name:50s}: {status}")

    working_configs = [name for name, passed in results.items() if passed]

    print("\n" + "=" * 80)
    if working_configs:
        print(f"✓ Found {len(working_configs)} working configuration(s):")
        for config_name in working_configs:
            print(f"  - {config_name}")
        print("\nApply the working configuration to scripts/train.py")
        return 0
    else:
        print("✗ No working NCCL configurations found")
        print("\nNext steps:")
        print("1. Check /dev/shm permissions and size: df -h /dev/shm")
        print("2. Check NCCL installation: ldconfig -p | grep nccl")
        print("3. Try using FSDP instead of DDP (Strategy 2.4)")
        return 1


if __name__ == "__main__":
    # Check we're in the right directory
    if not Path("tests/diagnostics/test_small_model_ddp.py").exists():
        print("ERROR: Must run from giblet-responses root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)

    sys.exit(main())
