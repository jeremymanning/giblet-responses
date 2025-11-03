#!/usr/bin/env python3
"""
NCCL Backend Health Diagnostic Test (Issue #30, Phase 1.1)

Tests NCCL availability, version, and basic functionality on the cluster.
This test runs on a SINGLE process to verify NCCL backend is properly configured.

Usage:
    python tests/diagnostics/test_nccl_health.py

Expected output:
    - NCCL availability status
    - NCCL version
    - Process group initialization success
    - nvlink status between GPUs
"""

import os
import sys
import subprocess
import torch
import torch.distributed as dist

def test_nccl_availability():
    """Test if NCCL backend is available."""
    print("=" * 80)
    print("Test 1: NCCL Availability")
    print("=" * 80)

    is_available = dist.is_nccl_available()
    print(f"NCCL available: {is_available}")

    if not is_available:
        print("ERROR: NCCL backend not available!")
        return False

    print("✓ NCCL backend is available")
    return True


def test_nccl_version():
    """Test NCCL version."""
    print("\n" + "=" * 80)
    print("Test 2: NCCL Version")
    print("=" * 80)

    try:
        version = torch.cuda.nccl.version()
        # Version may be returned as tuple (major, minor, patch) or int
        if isinstance(version, tuple):
            major, minor, patch = version
            print(f"NCCL version: {major}.{minor}.{patch}")
        else:
            major = version // 1000
            minor = (version % 1000) // 100
            patch = version % 100
            print(f"NCCL version: {major}.{minor}.{patch}")

        print("✓ NCCL version retrieved successfully")
        return True
    except Exception as e:
        print(f"ERROR: Could not get NCCL version: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda_devices():
    """Test CUDA device availability."""
    print("\n" + "=" * 80)
    print("Test 3: CUDA Devices")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False

    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")

    print(f"\n✓ Found {device_count} CUDA devices")
    return True


def test_process_group_init():
    """Test basic process group initialization (single process)."""
    print("\n" + "=" * 80)
    print("Test 4: Process Group Initialization (Single Process)")
    print("=" * 80)

    try:
        # Set environment variables for single-process init
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'

        print("Initializing process group with:")
        print(f"  Backend: nccl")
        print(f"  Rank: 0")
        print(f"  World size: 1")

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )

        print("✓ Process group initialized successfully")

        # Test basic operations
        print("\nTesting basic tensor operations...")
        device = torch.device('cuda:0')
        tensor = torch.randn(10, 10).to(device)
        print(f"  Created tensor on {device}: shape {tensor.shape}")

        # Cleanup
        dist.destroy_process_group()
        print("✓ Process group destroyed successfully")

        return True

    except Exception as e:
        print(f"ERROR: Process group initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nvlink_status():
    """Test nvlink connectivity between GPUs."""
    print("\n" + "=" * 80)
    print("Test 5: NVLink Status")
    print("=" * 80)

    try:
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("NVLink status:")
            print(result.stdout)
            print("✓ NVLink status retrieved successfully")
            return True
        else:
            print(f"WARNING: nvidia-smi nvlink returned error code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi nvlink command timed out")
        return False
    except FileNotFoundError:
        print("WARNING: nvidia-smi command not found (may not be in PATH)")
        return False
    except Exception as e:
        print(f"ERROR: Could not get nvlink status: {e}")
        return False


def test_p2p_access():
    """Test peer-to-peer GPU access."""
    print("\n" + "=" * 80)
    print("Test 6: Peer-to-Peer GPU Access")
    print("=" * 80)

    device_count = torch.cuda.device_count()

    if device_count < 2:
        print("INFO: Only 1 GPU available, skipping P2P test")
        return True

    print(f"Testing P2P access between {device_count} GPUs:")

    p2p_matrix = []
    for i in range(device_count):
        row = []
        for j in range(device_count):
            if i == j:
                row.append('N/A')
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                row.append('✓' if can_access else '✗')
        p2p_matrix.append(row)

    # Print matrix
    print("\nP2P Access Matrix (rows=source, cols=destination):")
    print("     ", end="")
    for j in range(device_count):
        print(f"GPU{j:2d} ", end="")
    print()

    for i in range(device_count):
        print(f"GPU{i:2d}: ", end="")
        for j in range(device_count):
            print(f" {p2p_matrix[i][j]:^4s}", end="")
        print()

    # Check if any P2P is enabled
    p2p_enabled = any(cell == '✓' for row in p2p_matrix for cell in row)

    if p2p_enabled:
        print("\n✓ P2P access enabled between some GPU pairs")
    else:
        print("\nWARNING: P2P access not enabled between any GPU pairs")
        print("This may impact multi-GPU performance")

    return True


def main():
    """Run all NCCL health diagnostic tests."""
    print("\n" + "=" * 80)
    print("NCCL Backend Health Diagnostic Test")
    print("Issue #30, Phase 1.1")
    print("=" * 80)

    results = {}

    # Run tests
    results['nccl_availability'] = test_nccl_availability()
    results['nccl_version'] = test_nccl_version()
    results['cuda_devices'] = test_cuda_devices()
    results['process_group_init'] = test_process_group_init()
    results['nvlink_status'] = test_nvlink_status()
    results['p2p_access'] = test_p2p_access()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests PASSED")
        print("NCCL backend is healthy and properly configured")
    else:
        print("✗ Some tests FAILED")
        print("NCCL backend may have configuration issues")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
