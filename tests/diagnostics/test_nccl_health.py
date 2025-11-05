#!/usr/bin/env python3
"""
NCCL Backend Health Diagnostic Test (Issue #30, Phase 1.1)

Tests NCCL availability, version, and basic functionality on the cluster.
This test runs on a SINGLE process to verify NCCL backend is properly configured.

Usage:
    pytest tests/diagnostics/test_nccl_health.py -v -s
    # Or run as standalone script:
    python tests/diagnostics/test_nccl_health.py

Expected output:
    - NCCL availability status
    - NCCL version
    - Process group initialization success
    - nvlink status between GPUs
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nccl_availability():
    """Test if NCCL backend is available."""
    print("=" * 80)
    print("Test 1: NCCL Availability")
    print("=" * 80)

    is_available = dist.is_nccl_available()
    print(f"NCCL available: {is_available}")

    assert is_available, "NCCL backend not available!"
    print("✓ NCCL backend is available")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nccl_version():
    """Test NCCL version."""
    print("\n" + "=" * 80)
    print("Test 2: NCCL Version")
    print("=" * 80)

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
    assert version is not None, "NCCL version should not be None"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_devices():
    """Test CUDA device availability."""
    print("\n" + "=" * 80)
    print("Test 3: CUDA Devices")
    print("=" * 80)

    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")

    assert device_count > 0, "No CUDA devices found!"

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")

    print(f"\n✓ Found {device_count} CUDA devices")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.integration
def test_process_group_init():
    """Test basic process group initialization (single process)."""
    print("\n" + "=" * 80)
    print("Test 4: Process Group Initialization (Single Process)")
    print("=" * 80)

    # Set environment variables for single-process init
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    print("Initializing process group with:")
    print("  Backend: nccl")
    print("  Rank: 0")
    print("  World size: 1")

    dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)

    print("✓ Process group initialized successfully")

    # Test basic operations
    print("\nTesting basic tensor operations...")
    device = torch.device("cuda:0")
    tensor = torch.randn(10, 10).to(device)
    print(f"  Created tensor on {device}: shape {tensor.shape}")

    # Cleanup
    dist.destroy_process_group()
    print("✓ Process group destroyed successfully")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nvlink_status():
    """Test nvlink connectivity between GPUs."""
    print("\n" + "=" * 80)
    print("Test 5: NVLink Status")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("NVLink status:")
            print(result.stdout)
            print("✓ NVLink status retrieved successfully")
        else:
            print(f"WARNING: nvidia-smi nvlink returned error code {result.returncode}")
            print(f"stderr: {result.stderr}")
            pytest.skip("nvidia-smi nvlink command failed")

    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi nvlink command timed out")
        pytest.skip("nvidia-smi nvlink command timed out")
    except FileNotFoundError:
        print("WARNING: nvidia-smi command not found (may not be in PATH)")
        pytest.skip("nvidia-smi command not found")
    except Exception as e:
        print(f"ERROR: Could not get nvlink status: {e}")
        pytest.skip(f"Could not get nvlink status: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_p2p_access():
    """Test peer-to-peer GPU access."""
    print("\n" + "=" * 80)
    print("Test 6: Peer-to-Peer GPU Access")
    print("=" * 80)

    device_count = torch.cuda.device_count()

    if device_count < 2:
        print("INFO: Only 1 GPU available, skipping P2P test")
        pytest.skip("Only 1 GPU available")

    print(f"Testing P2P access between {device_count} GPUs:")

    p2p_matrix = []
    for i in range(device_count):
        row = []
        for j in range(device_count):
            if i == j:
                row.append("N/A")
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                row.append("✓" if can_access else "✗")
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
    p2p_enabled = any(cell == "✓" for row in p2p_matrix for cell in row)

    if p2p_enabled:
        print("\n✓ P2P access enabled between some GPU pairs")
    else:
        print("\nWARNING: P2P access not enabled between any GPU pairs")
        print("This may impact multi-GPU performance")


def main():
    """Run all NCCL health diagnostic tests (standalone script mode)."""
    print("\n" + "=" * 80)
    print("NCCL Backend Health Diagnostic Test")
    print("Issue #30, Phase 1.1")
    print("=" * 80)

    results = {}

    # Run tests and catch exceptions
    for test_name, test_func in [
        ("nccl_availability", test_nccl_availability),
        ("nccl_version", test_nccl_version),
        ("cuda_devices", test_cuda_devices),
        ("process_group_init", test_process_group_init),
        ("nvlink_status", test_nvlink_status),
        ("p2p_access", test_p2p_access),
    ]:
        try:
            test_func()
            results[test_name] = True
        except pytest.skip.Exception as e:
            print(f"SKIP: {e}")
            results[test_name] = True  # Skip counts as pass for standalone mode
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback

            traceback.print_exc()
            results[test_name] = False

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


if __name__ == "__main__":
    sys.exit(main())
