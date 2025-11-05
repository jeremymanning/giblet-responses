#!/usr/bin/env python3
"""
Pre-generate cache file for all 17 subjects.

This script loads the multimodal dataset with all 17 subjects to trigger
preprocessing and caching, avoiding NCCL timeout issues during distributed
training initialization.

Usage:
    python scripts/pregenerate_cache.py --config CONFIG_FILE
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from giblet.data.dataset import MultimodalDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Pre-generate dataset cache")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config YAML file"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Pre-generating Dataset Cache")
    print("=" * 80)
    print(f"Config file: {config_path}")
    print()

    # Extract dataset parameters from config
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    dataset_params = {
        "data_dir": data_config.get("data_dir", "data/"),
        "subjects": data_config.get("subjects", "all"),
        "split": None,  # Load full dataset for caching
        "apply_hrf": data_config.get("apply_hrf", True),
        "mode": data_config.get("mode", "per_subject"),
        "tr": data_config.get("tr", 1.5),
        "use_encodec": model_config.get("use_encodec", True),
        "encodec_bandwidth": 3.0,
        "encodec_sample_rate": 12000,
        "frame_skip": data_config.get("frame_skip", 2),
        "preprocess": True,  # Force preprocessing
    }

    print("Dataset parameters:")
    for key, value in dataset_params.items():
        print(f"  {key}: {value}")
    print()

    # Create dataset (this will trigger preprocessing and caching)
    print("Creating dataset (this will preprocess and cache all data)...")
    print("This may take 10-20 minutes for all 17 subjects.")
    print()

    dataset = MultimodalDataset(**dataset_params)

    print()
    print("=" * 80)
    print("Cache Generation Complete")
    print("=" * 80)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Feature dimensions: {dataset.feature_dims}")
    print()
    print("You can now launch distributed training without timeout issues.")
    print("=" * 80)


if __name__ == "__main__":
    main()
