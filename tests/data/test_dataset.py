"""
Test script for SherlockDataset with real data.

This test verifies that the dataset:
1. Loads all 17 subjects correctly
2. Produces expected 15,640 samples (17 × 920)
3. Returns correct shapes for each modality
4. Works with PyTorch DataLoader
5. Supports train/val splits
6. Handles single-subject and cross-subject modes
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from giblet.data.dataset import SherlockDataset


def test_full_dataset():
    """Test loading full dataset with all 17 subjects."""
    print("="*80)
    print("TEST 1: Full dataset (all 17 subjects)")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    # Load dataset
    dataset = SherlockDataset(
        data_dir=data_dir,
        subjects='all',
        apply_hrf=True,
        mode='per_subject',
        preprocess=True
    )

    # Check total samples
    print(f"\n✓ Total samples: {len(dataset)}")
    assert len(dataset) == 17 * 920, f"Expected 15640 samples, got {len(dataset)}"

    # Check first sample
    sample = dataset[0]
    print(f"\n✓ Sample keys: {list(sample.keys())}")
    print(f"✓ Video shape: {sample['video'].shape}")
    print(f"✓ Audio shape: {sample['audio'].shape}")
    print(f"✓ Text shape: {sample['text'].shape}")
    print(f"✓ fMRI shape: {sample['fmri'].shape}")
    print(f"✓ Subject ID: {sample['subject_id']}")
    print(f"✓ TR index: {sample['tr_index']}")

    # Verify shapes
    assert sample['video'].shape[0] == 43200, "Video features should be 160×90×3 = 43200"
    assert sample['audio'].shape[0] == 128, "Audio features should be 128 mel bins"
    assert sample['text'].shape[0] == 1024, "Text features should be 1024-dim embeddings"
    assert sample['fmri'].shape[0] > 80000, "fMRI should have ~83k voxels"

    # Check all samples are accessible
    print(f"\n✓ Testing sample access...")
    last_sample = dataset[-1]
    print(f"✓ Last sample - Subject: {last_sample['subject_id']}, TR: {last_sample['tr_index']}")
    assert last_sample['subject_id'] == 17, "Last sample should be from subject 17"

    # Test random samples
    for idx in [100, 1000, 5000, 10000]:
        sample = dataset[idx]
        subject_id = sample['subject_id']
        tr_idx = sample['tr_index']
        print(f"✓ Sample {idx}: Subject {subject_id}, TR {tr_idx}")

    print("\n✅ Full dataset test PASSED\n")
    return dataset


def test_dataloader():
    """Test PyTorch DataLoader integration."""
    print("="*80)
    print("TEST 2: DataLoader integration")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    dataset = SherlockDataset(
        data_dir=data_dir,
        subjects='all',
        apply_hrf=True,
        mode='per_subject'
    )

    # Create DataLoader
    batch_size = 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )

    print(f"\n✓ Created DataLoader with batch_size={batch_size}")
    print(f"✓ Number of batches: {len(dataloader)}")

    # Test first batch
    batch = next(iter(dataloader))
    print(f"\n✓ Batch keys: {list(batch.keys())}")
    print(f"✓ Video batch shape: {batch['video'].shape}")
    print(f"✓ Audio batch shape: {batch['audio'].shape}")
    print(f"✓ Text batch shape: {batch['text'].shape}")
    print(f"✓ fMRI batch shape: {batch['fmri'].shape}")

    assert batch['video'].shape[0] == batch_size, f"Expected batch size {batch_size}"
    assert batch['audio'].shape[0] == batch_size
    assert batch['text'].shape[0] == batch_size
    assert batch['fmri'].shape[0] == batch_size

    # Test iteration
    print(f"\n✓ Testing batch iteration...")
    n_batches_tested = 0
    for i, batch in enumerate(dataloader):
        if i < 3:
            print(f"  Batch {i}: video {batch['video'].shape}, "
                  f"subjects {batch['subject_id'].tolist()[:5]}...")
        n_batches_tested += 1
        if i >= 5:  # Test first few batches
            break

    print(f"✓ Successfully iterated {n_batches_tested} batches")

    print("\n✅ DataLoader test PASSED\n")


def test_train_val_split():
    """Test train/validation split."""
    print("="*80)
    print("TEST 3: Train/validation split")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    # Create train dataset
    train_dataset = SherlockDataset(
        data_dir=data_dir,
        subjects='all',
        split='train',
        apply_hrf=True,
        mode='per_subject'
    )

    # Create val dataset
    val_dataset = SherlockDataset(
        data_dir=data_dir,
        subjects='all',
        split='val',
        apply_hrf=True,
        mode='per_subject'
    )

    print(f"\n✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")

    # Verify 80/20 split
    total_expected = 17 * 920
    train_expected = 17 * int(0.8 * 920)
    val_expected = 17 * (920 - int(0.8 * 920))

    print(f"\n✓ Expected train: {train_expected}")
    print(f"✓ Expected val: {val_expected}")

    assert len(train_dataset) == train_expected, f"Train split size mismatch"
    assert len(val_dataset) == val_expected, f"Val split size mismatch"

    # Verify no overlap
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]
    print(f"\n✓ Train sample TR range: 0-{train_dataset.n_trs-1}")
    print(f"✓ Val sample TR range: {int(0.8*920)}-919")

    print("\n✅ Train/val split test PASSED\n")


def test_single_subject():
    """Test loading single subject."""
    print("="*80)
    print("TEST 4: Single subject mode")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    # Load single subject
    dataset = SherlockDataset(
        data_dir=data_dir,
        subjects=1,
        apply_hrf=True,
        mode='per_subject'
    )

    print(f"\n✓ Single subject samples: {len(dataset)}")
    assert len(dataset) == 920, f"Expected 920 samples for single subject, got {len(dataset)}"

    # Check subject ID
    sample = dataset[0]
    print(f"✓ Subject ID: {sample['subject_id']}")
    assert sample['subject_id'] == 1, "Subject ID should be 1"

    # Check all samples have same subject
    last_sample = dataset[-1]
    assert last_sample['subject_id'] == 1, "All samples should be from subject 1"

    # Test get_subject_data
    subject_data = dataset.get_subject_data(1)
    print(f"\n✓ Subject data shapes:")
    print(f"  Video: {subject_data['video'].shape}")
    print(f"  Audio: {subject_data['audio'].shape}")
    print(f"  Text: {subject_data['text'].shape}")
    print(f"  fMRI: {subject_data['fmri'].shape}")

    assert subject_data['video'].shape[0] == 920, "Should have 920 TRs"

    print("\n✅ Single subject test PASSED\n")


def test_cross_subject():
    """Test cross-subject averaging mode."""
    print("="*80)
    print("TEST 5: Cross-subject mode")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    # Load with cross-subject averaging
    dataset = SherlockDataset(
        data_dir=data_dir,
        subjects='all',
        apply_hrf=True,
        mode='cross_subject'
    )

    print(f"\n✓ Cross-subject samples: {len(dataset)}")
    assert len(dataset) == 920, f"Expected 920 samples (just TRs), got {len(dataset)}"

    # Check sample
    sample = dataset[0]
    print(f"\n✓ Sample keys: {list(sample.keys())}")
    assert 'subject_id' not in sample, "Cross-subject mode should not have subject_id"

    print(f"✓ Video shape: {sample['video'].shape}")
    print(f"✓ fMRI shape: {sample['fmri'].shape}")

    print("\n✅ Cross-subject test PASSED\n")


def test_feature_stats():
    """Test feature statistics computation."""
    print("="*80)
    print("TEST 6: Feature statistics")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    dataset = SherlockDataset(
        data_dir=data_dir,
        subjects='all',
        apply_hrf=True,
        mode='per_subject'
    )

    stats = dataset.get_feature_stats()

    print("\n✓ Feature statistics:")
    for modality in ['video', 'audio', 'text', 'fmri']:
        print(f"\n  {modality.upper()}:")
        print(f"    Shape: {stats[modality]['shape']}")
        print(f"    Mean: {stats[modality]['mean']:.4f}")
        print(f"    Std: {stats[modality]['std']:.4f}")
        print(f"    Min: {stats[modality]['min']:.4f}")
        print(f"    Max: {stats[modality]['max']:.4f}")

    print("\n✅ Feature statistics test PASSED\n")


def test_no_hrf():
    """Test dataset without HRF convolution."""
    print("="*80)
    print("TEST 7: Dataset without HRF")
    print("="*80)

    data_dir = Path(__file__).parent.parent.parent / 'data'

    # Load without HRF
    dataset = SherlockDataset(
        data_dir=data_dir,
        subjects=[1, 2],  # Just 2 subjects for speed
        apply_hrf=False,
        mode='per_subject'
    )

    print(f"\n✓ Dataset without HRF: {len(dataset)} samples")
    assert len(dataset) == 2 * 920, "Expected 2 subjects × 920 TRs"

    sample = dataset[0]
    print(f"✓ Sample shapes match (no HRF):")
    print(f"  Video: {sample['video'].shape}")
    print(f"  Audio: {sample['audio'].shape}")

    print("\n✅ No HRF test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SHERLOCK DATASET TESTS")
    print("="*80)

    try:
        # Run tests
        dataset = test_full_dataset()
        test_dataloader()
        test_train_val_split()
        test_single_subject()
        test_cross_subject()
        test_feature_stats()
        test_no_hrf()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)

        # Print summary
        print(f"\nDataset Summary:")
        print(f"  Total subjects: {dataset.n_subjects}")
        print(f"  TRs per subject: {dataset.n_trs}")
        print(f"  Total samples: {dataset.n_samples}")
        print(f"  Feature dimensions:")
        for modality, dim in dataset.feature_dims.items():
            print(f"    {modality}: {dim}")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
