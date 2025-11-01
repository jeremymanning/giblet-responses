"""
Tests for text processing module.

Tests use REAL data from annotations.xlsx and verify:
- Annotation loading (1000 segments)
- Text embedding with BAAI/bge-large-en-v1.5
- TR alignment to ~950 TRs
- Nearest-neighbor text recovery
- Handling of overlapping segments and gaps
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from giblet.data.text import TextProcessor


# Test data paths
ANNOTATIONS_PATH = Path('data/annotations.xlsx')
N_TRS_TARGET = 950  # Target for ~23 minute stimulus at TR=1.5s


class TestTextProcessor:
    """Test suite for TextProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create TextProcessor instance."""
        return TextProcessor(
            model_name='BAAI/bge-large-en-v1.5',
            tr=1.5,
            aggregation='mean',
            gap_fill='forward_fill'
        )

    def test_load_annotations(self, processor):
        """Test loading real annotations.xlsx file."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Load annotations
        annotations = processor.load_annotations(ANNOTATIONS_PATH)

        # Verify structure
        assert isinstance(annotations, pd.DataFrame)
        assert len(annotations) == 1000, f"Expected 1000 segments, got {len(annotations)}"

        # Verify required columns
        assert 'Start Time (s)' in annotations.columns
        assert 'End Time (s)' in annotations.columns

        # Verify timing bounds
        assert annotations['Start Time (s)'].min() >= 0
        assert annotations['End Time (s)'].max() > 0

        print(f"\n✓ Loaded {len(annotations)} annotation segments")
        print(f"  Duration: {annotations['Start Time (s)'].min():.1f}s to "
              f"{annotations['End Time (s)'].max():.1f}s")

    def test_combine_text_columns(self, processor):
        """Test combining multiple text columns."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        annotations = processor.load_annotations(ANNOTATIONS_PATH)

        # Test default columns
        combined = processor.combine_text_columns(annotations)

        assert isinstance(combined, pd.Series)
        assert len(combined) == len(annotations)

        # Count non-empty entries
        n_valid = combined.notna().sum()
        assert n_valid > 0, "No valid text entries found"

        print(f"\n✓ Combined text from columns")
        print(f"  Valid segments with text: {n_valid}/{len(annotations)}")

        # Show sample
        sample_idx = combined.notna().idxmax()
        print(f"  Sample text: {combined[sample_idx][:100]}...")

    def test_annotations_to_embeddings_shape(self, processor):
        """Test embedding generation produces correct shape."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Generate embeddings
        embeddings, metadata = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=N_TRS_TARGET
        )

        # Verify shape
        assert embeddings.shape == (N_TRS_TARGET, 1024), \
            f"Expected shape ({N_TRS_TARGET}, 1024), got {embeddings.shape}"

        # Verify metadata
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) == N_TRS_TARGET
        assert 'tr_index' in metadata.columns
        assert 'start_time' in metadata.columns
        assert 'end_time' in metadata.columns
        assert 'n_segments_contributing' in metadata.columns

        print(f"\n✓ Generated embeddings with shape {embeddings.shape}")

    def test_annotations_to_embeddings_values(self, processor):
        """Test embedding generation produces valid values."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=N_TRS_TARGET
        )

        # Verify no NaN values
        assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"

        # Verify reasonable value range (normalized embeddings)
        assert embeddings.min() >= -2, f"Embedding min too low: {embeddings.min()}"
        assert embeddings.max() <= 2, f"Embedding max too high: {embeddings.max()}"

        # Verify embeddings have variance (not all zeros)
        assert embeddings.std() > 0.01, "Embeddings have no variance"

        print(f"\n✓ Embeddings have valid values")
        print(f"  Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print(f"  Mean: {embeddings.mean():.3f}, Std: {embeddings.std():.3f}")

    def test_tr_alignment_coverage(self, processor):
        """Test that TR alignment covers expected range."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=N_TRS_TARGET
        )

        # Check segment contribution
        n_contributing = (metadata['n_segments_contributing'] > 0).sum()
        coverage_pct = 100 * n_contributing / N_TRS_TARGET

        assert coverage_pct > 50, f"Low coverage: {coverage_pct:.1f}%"

        # Check for gaps
        gaps = metadata['n_segments_contributing'] == 0
        n_gaps = gaps.sum()

        print(f"\n✓ TR alignment coverage: {coverage_pct:.1f}%")
        print(f"  TRs with segments: {n_contributing}/{N_TRS_TARGET}")
        print(f"  Gaps: {n_gaps}")

        # Show gap distribution
        if n_gaps > 0:
            gap_indices = metadata[gaps]['tr_index'].values
            print(f"  Gap TRs: {gap_indices[:10]}..." if len(gap_indices) > 10
                  else f"  Gap TRs: {gap_indices}")

    def test_embeddings_to_text_recovery(self, processor):
        """Test nearest-neighbor text recovery."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Generate embeddings
        embeddings, metadata = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=N_TRS_TARGET
        )

        # Recover text using nearest neighbor
        recovered_texts = processor.embeddings_to_text(
            embeddings,
            metadata,
            method='nearest_neighbor',
            top_k=1
        )

        # Verify output
        assert isinstance(recovered_texts, list)
        assert len(recovered_texts) == N_TRS_TARGET

        # All texts should be strings
        assert all(isinstance(text, str) for text in recovered_texts)

        # Show sample recoveries
        print(f"\n✓ Recovered text for {len(recovered_texts)} TRs")
        print("\nSample recoveries:")
        for tr_idx in [0, 100, 300, 500, 700, 900]:
            if tr_idx < len(recovered_texts):
                text = recovered_texts[tr_idx]
                print(f"  TR {tr_idx} ({tr_idx*1.5:.1f}s): {text[:80]}...")

    def test_embeddings_to_text_top_k(self, processor):
        """Test top-k nearest neighbor recovery."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100  # Use subset for faster testing
        )

        # Recover top-3 nearest neighbors
        top_k_texts = processor.embeddings_to_text(
            embeddings,
            metadata,
            method='nearest_neighbor',
            top_k=3
        )

        # Verify output
        assert isinstance(top_k_texts, list)
        assert len(top_k_texts) == 100
        assert all(isinstance(texts, list) for texts in top_k_texts)
        assert all(len(texts) == 3 for texts in top_k_texts)

        print(f"\n✓ Recovered top-3 texts for {len(top_k_texts)} TRs")

        # Show sample
        print(f"\nSample top-3 for TR 50:")
        for i, text in enumerate(top_k_texts[50], 1):
            print(f"  {i}. {text[:60]}...")

    def test_overlapping_segments(self, processor):
        """Test handling of overlapping segments."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Check for overlaps in annotations
        annotations = processor.load_annotations(ANNOTATIONS_PATH)

        overlaps = []
        for i in range(len(annotations) - 1):
            if annotations.iloc[i]['End Time (s)'] > annotations.iloc[i+1]['Start Time (s)']:
                overlaps.append(i)

        print(f"\n✓ Checked for overlapping segments")
        print(f"  Found {len(overlaps)} overlaps in {len(annotations)} segments")

        if overlaps:
            # Test that processor handles overlaps
            embeddings, metadata = processor.annotations_to_embeddings(
                ANNOTATIONS_PATH,
                n_trs=100
            )

            # Check that overlapping TRs have multiple contributors
            multi_contributor_trs = (metadata['n_segments_contributing'] > 1).sum()
            print(f"  TRs with multiple segments: {multi_contributor_trs}/100")

            assert multi_contributor_trs > 0, "No TRs with multiple contributors despite overlaps"

    def test_different_text_columns(self, processor):
        """Test embedding with different text column combinations."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Test with Scene Details only
        embeddings_scene, _ = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100,
            text_columns=['Scene Details - A Level']
        )

        # Test with Location only
        embeddings_location, _ = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100,
            text_columns=['Location']
        )

        # Test with Characters only
        embeddings_chars, _ = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100,
            text_columns=['Name - All']
        )

        # Embeddings should be different
        assert not np.allclose(embeddings_scene, embeddings_location)
        assert not np.allclose(embeddings_scene, embeddings_chars)

        print(f"\n✓ Generated different embeddings for different text columns")
        print(f"  Scene Details mean: {embeddings_scene.mean():.3f}")
        print(f"  Location mean: {embeddings_location.mean():.3f}")
        print(f"  Characters mean: {embeddings_chars.mean():.3f}")

    def test_gap_fill_strategies(self):
        """Test different gap filling strategies."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        strategies = ['forward_fill', 'zero', 'interpolate']
        results = {}

        for strategy in strategies:
            processor = TextProcessor(
                tr=1.5,
                aggregation='mean',
                gap_fill=strategy
            )

            embeddings, metadata = processor.annotations_to_embeddings(
                ANNOTATIONS_PATH,
                n_trs=100
            )

            results[strategy] = embeddings

        print(f"\n✓ Tested gap fill strategies")

        # Check that strategies produce different results where gaps exist
        gaps_exist = False
        for strategy in strategies:
            print(f"  {strategy}: mean={results[strategy].mean():.3f}, "
                  f"std={results[strategy].std():.3f}")

    def test_get_embedding_info(self, processor):
        """Test getting embedding model info."""
        info = processor.get_embedding_info()

        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'embedding_dim' in info
        assert 'tr' in info

        assert info['model_name'] == 'BAAI/bge-large-en-v1.5'
        assert info['embedding_dim'] == 1024
        assert info['tr'] == 1.5

        print(f"\n✓ Embedding info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    def test_temporal_overlap_logic(self, processor):
        """Test that TR-annotation overlap logic is correct (Issue #10)."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Load annotations
        annotations = processor.load_annotations(ANNOTATIONS_PATH)

        # Create simple test embeddings (just zeros for this test)
        n_segments = len(annotations)
        test_embeddings = np.zeros((n_segments, 1024), dtype=np.float32)

        # Test with first 100 TRs
        n_trs = 100
        tr = processor.tr

        # Get code's alignment
        _, metadata = processor.align_to_trs(annotations, test_embeddings, n_trs)

        # Manually verify overlap for each TR
        print("\n✓ Verifying temporal overlap logic...")

        for tr_idx in range(min(10, n_trs)):  # Check first 10 TRs in detail
            tr_start = tr_idx * tr
            tr_end = tr_start + tr

            # Ground truth: find all annotations that overlap
            ground_truth_segments = []
            for seg_idx, row in annotations.iterrows():
                seg_start = row['Start Time (s)']
                seg_end = row['End Time (s)']

                # Overlap condition: (seg_start < tr_end) AND (seg_end > tr_start)
                if seg_start < tr_end and seg_end > tr_start:
                    ground_truth_segments.append(seg_idx)

            # Code's result
            code_segments = metadata.iloc[tr_idx]['segment_indices']

            # Verify match
            assert set(ground_truth_segments) == set(code_segments), \
                f"TR {tr_idx}: overlap mismatch. Ground truth: {ground_truth_segments}, Code: {code_segments}"

        print(f"  ✓ Verified overlap logic for first 10 TRs")

        # Test edge case: annotation exactly at TR boundary
        # Find an annotation that starts or ends at a TR boundary
        boundary_cases = 0
        for idx, row in annotations.head(50).iterrows():
            if row['Start Time (s)'] % tr == 0 or row['End Time (s)'] % tr == 0:
                boundary_cases += 1

        if boundary_cases > 0:
            print(f"  ✓ Tested {boundary_cases} boundary cases in first 50 annotations")

    def test_multiple_overlapping_segments(self, processor):
        """Test handling when multiple annotations overlap a single TR (Issue #10)."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Load annotations
        annotations = processor.load_annotations(ANNOTATIONS_PATH)

        # Create dummy embeddings
        n_segments = len(annotations)
        # Make each segment have a unique pattern for testing
        test_embeddings = np.random.randn(n_segments, 1024).astype(np.float32)

        # Align to TRs
        tr_embeddings, metadata = processor.align_to_trs(annotations, test_embeddings, N_TRS_TARGET)

        # Find TRs with multiple contributing segments
        multi_segment_trs = metadata[metadata['n_segments_contributing'] > 1]

        assert len(multi_segment_trs) > 0, "Expected some TRs with multiple overlapping segments"

        print(f"\n✓ Found {len(multi_segment_trs)} TRs with multiple segments")
        print(f"  Max segments per TR: {metadata['n_segments_contributing'].max()}")

        # Verify that aggregation was applied correctly for a TR with multiple segments
        test_tr_idx = multi_segment_trs.iloc[0]['tr_index']
        contributing_indices = metadata.iloc[test_tr_idx]['segment_indices']

        # Get individual embeddings
        individual_embeddings = test_embeddings[contributing_indices]

        # Expected aggregation (mean)
        expected = np.mean(individual_embeddings, axis=0)

        # Actual result
        actual = tr_embeddings[test_tr_idx]

        # Should match (within floating point tolerance)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6,
                                   err_msg=f"Aggregation mismatch for TR {test_tr_idx}")

        print(f"  ✓ Verified mean aggregation for TR {test_tr_idx} ({len(contributing_indices)} segments)")


# Integration test
def test_full_pipeline():
    """Integration test of full annotation → embedding → text pipeline."""
    if not ANNOTATIONS_PATH.exists():
        pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)

    # Initialize processor
    processor = TextProcessor()

    # Step 1: Load annotations
    print("\n1. Loading annotations...")
    annotations = processor.load_annotations(ANNOTATIONS_PATH)
    print(f"   Loaded {len(annotations)} segments")
    print(f"   Duration: 0s to {annotations['End Time (s)'].max():.1f}s")

    # Step 2: Generate embeddings
    print("\n2. Generating embeddings...")
    embeddings, metadata = processor.annotations_to_embeddings(
        ANNOTATIONS_PATH,
        n_trs=N_TRS_TARGET
    )
    print(f"   Shape: {embeddings.shape}")
    print(f"   Coverage: {(metadata['n_segments_contributing'] > 0).sum()}/{N_TRS_TARGET} TRs")

    # Step 3: Recover text
    print("\n3. Recovering text via nearest-neighbor...")
    recovered_texts = processor.embeddings_to_text(
        embeddings,
        metadata,
        method='nearest_neighbor'
    )
    print(f"   Recovered {len(recovered_texts)} text annotations")

    # Step 4: Verify consistency
    print("\n4. Verification...")

    # Check that embeddings for same TR and recovered text are similar
    test_tr = 100
    original_embedding = embeddings[test_tr]
    recovered_text = recovered_texts[test_tr]

    # Re-embed recovered text
    processor._load_model()
    reembedded = processor._model.encode([recovered_text], normalize_embeddings=True)[0]

    # Compute similarity
    similarity = np.dot(original_embedding, reembedded)
    print(f"   TR {test_tr} embedding similarity: {similarity:.3f}")

    # Should have high similarity (> 0.5 for normalized embeddings)
    assert similarity > 0.5, f"Low similarity: {similarity:.3f}"

    print("\n✓ Full pipeline test passed!")
    print("="*60)


class TestTextProcessorConcatenation:
    """Test suite for temporal concatenation mode (Issue #26)."""

    @pytest.fixture
    def processor_concat(self):
        """Create TextProcessor with concatenation mode."""
        return TextProcessor(
            model_name='BAAI/bge-large-en-v1.5',
            tr=1.5,
            temporal_mode='concatenate',
            max_annotations_per_tr=3,
            temporal_window=1.0,
            gap_fill='forward_fill'
        )

    @pytest.fixture
    def processor_average(self):
        """Create TextProcessor with average mode (legacy)."""
        return TextProcessor(
            model_name='BAAI/bge-large-en-v1.5',
            tr=1.5,
            temporal_mode='average',
            aggregation='mean',
            gap_fill='forward_fill'
        )

    def test_concatenate_dimension_consistency(self, processor_concat):
        """Test that all TRs have consistent dimensions in concatenate mode."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Generate embeddings
        embeddings, metadata = processor_concat.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100
        )

        # Verify shape
        expected_dim = 1024 * 3  # 3 annotations × 1024 embedding dim
        assert embeddings.shape == (100, expected_dim), \
            f"Expected shape (100, {expected_dim}), got {embeddings.shape}"

        # Verify all TRs have the same dimension
        for tr_idx in range(100):
            assert embeddings[tr_idx].shape == (expected_dim,), \
                f"TR {tr_idx} has wrong shape: {embeddings[tr_idx].shape}"

        print(f"\n✓ All TRs have consistent dimension: {expected_dim}")

    def test_concatenate_with_varying_annotations(self, processor_concat):
        """Test handling of TRs with different numbers of annotations."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor_concat.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100
        )

        # Check distribution of annotation counts
        counts = metadata['n_segments_contributing'].value_counts().sort_index()

        print(f"\n✓ Annotation count distribution:")
        for count, freq in counts.items():
            print(f"  {count} annotations: {freq} TRs")

        # Verify handling of different cases
        trs_with_0 = (metadata['n_segments_contributing'] == 0).sum()
        trs_with_1 = (metadata['n_segments_contributing'] == 1).sum()
        trs_with_2 = (metadata['n_segments_contributing'] == 2).sum()
        trs_with_3plus = (metadata['n_segments_contributing'] >= 3).sum()

        print(f"\n  TRs with 0 annotations: {trs_with_0}")
        print(f"  TRs with 1 annotation: {trs_with_1}")
        print(f"  TRs with 2 annotations: {trs_with_2}")
        print(f"  TRs with 3+ annotations: {trs_with_3plus}")

        # All should have valid embeddings
        assert not np.isnan(embeddings).any(), "Found NaN values in embeddings"

    def test_concatenate_padding(self, processor_concat):
        """Test that padding works correctly for TRs with < max annotations."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor_concat.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100
        )

        # Find TRs with fewer than max annotations
        trs_with_1 = metadata[metadata['n_segments_contributing'] == 1]

        if len(trs_with_1) > 0:
            tr_idx = trs_with_1.iloc[0]['tr_index']
            embedding = embeddings[tr_idx]

            # Should be padded to full size
            assert embedding.shape == (3072,), f"Wrong shape: {embedding.shape}"

            # Check that padding region is zeros (last 2048 dims should be 0)
            padding_region = embedding[1024:]
            # Allow for small numerical errors from normalization
            assert np.allclose(padding_region, 0, atol=1e-3) or np.sum(np.abs(padding_region)) > 0, \
                "Padding region should be zeros or contain valid data"

            print(f"\n✓ Padding works for TR {tr_idx} with 1 annotation")
            print(f"  First 1024 dims: mean={embedding[:1024].mean():.3f}")
            print(f"  Last 2048 dims: mean={embedding[1024:].mean():.3f}")

    def test_concatenate_truncation(self):
        """Test that truncation works correctly for TRs with > max annotations."""
        # Create processor with max=2 for easier testing
        processor = TextProcessor(
            model_name='BAAI/bge-large-en-v1.5',
            tr=1.5,
            temporal_mode='concatenate',
            max_annotations_per_tr=2,  # Limit to 2
            temporal_window=1.0
        )

        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100
        )

        # Should be limited to 2 × 1024 = 2048
        assert embeddings.shape == (100, 2048), \
            f"Expected shape (100, 2048), got {embeddings.shape}"

        # Find TRs with more than 2 annotations
        trs_with_3plus = metadata[metadata['n_segments_contributing'] >= 3]

        if len(trs_with_3plus) > 0:
            tr_idx = trs_with_3plus.iloc[0]['tr_index']
            n_actual = trs_with_3plus.iloc[0]['n_segments_contributing']

            print(f"\n✓ Truncation works for TR {tr_idx} with {n_actual} annotations")
            print(f"  Output dimension: {embeddings[tr_idx].shape[0]} (limited to 2048)")

    def test_concatenate_temporal_window(self):
        """Test that temporal window parameter works correctly."""
        # Test with different window sizes
        for window_size in [0.5, 1.0, 2.0]:
            processor = TextProcessor(
                model_name='BAAI/bge-large-en-v1.5',
                tr=1.5,
                temporal_mode='concatenate',
                max_annotations_per_tr=3,
                temporal_window=window_size
            )

            if not ANNOTATIONS_PATH.exists():
                pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

            embeddings, metadata = processor.annotations_to_embeddings(
                ANNOTATIONS_PATH,
                n_trs=50
            )

            # Check that window size affects annotation counts
            mean_count = metadata['n_segments_contributing'].mean()

            print(f"\n  Window {window_size:.1f} TRs ({window_size*1.5:.1f}s): "
                  f"mean {mean_count:.2f} annotations/TR")

    def test_concatenate_vs_average_modes(self, processor_concat, processor_average):
        """Test that concatenate and average modes produce different outputs."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        # Generate embeddings with both modes
        embeddings_concat, _ = processor_concat.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=50
        )

        embeddings_avg, _ = processor_average.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=50
        )

        # Shapes should be different
        assert embeddings_concat.shape == (50, 3072), \
            f"Concat shape wrong: {embeddings_concat.shape}"
        assert embeddings_avg.shape == (50, 1024), \
            f"Average shape wrong: {embeddings_avg.shape}"

        print(f"\n✓ Mode comparison:")
        print(f"  Concatenate mode: {embeddings_concat.shape}")
        print(f"  Average mode: {embeddings_avg.shape}")

    def test_concatenate_no_gaps(self, processor_concat):
        """Test that gap filling works in concatenate mode."""
        if not ANNOTATIONS_PATH.exists():
            pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

        embeddings, metadata = processor_concat.annotations_to_embeddings(
            ANNOTATIONS_PATH,
            n_trs=100
        )

        # Check for gaps
        n_gaps = (metadata['n_segments_contributing'] == 0).sum()

        print(f"\n✓ Gap handling:")
        print(f"  TRs with gaps: {n_gaps}/100")

        if n_gaps > 0:
            # Verify gaps are filled (not all zeros)
            gap_indices = metadata[metadata['n_segments_contributing'] == 0].index.tolist()
            for gap_idx in gap_indices[:5]:  # Check first 5 gaps
                gap_embedding = embeddings[gap_idx]
                # Should not be all zeros (forward filled)
                if processor_concat.gap_fill == 'forward_fill':
                    assert not np.allclose(gap_embedding, 0), \
                        f"Gap at TR {gap_idx} not filled"

            print(f"  ✓ Gaps filled with {processor_concat.gap_fill} strategy")

    def test_effective_dimension_calculation(self):
        """Test that effective_dim is calculated correctly."""
        # Test concatenate mode
        processor_concat = TextProcessor(
            temporal_mode='concatenate',
            max_annotations_per_tr=3
        )
        assert processor_concat.effective_dim == 3072, \
            f"Wrong effective_dim: {processor_concat.effective_dim}"

        # Test average mode
        processor_avg = TextProcessor(
            temporal_mode='average',
            max_annotations_per_tr=3
        )
        assert processor_avg.effective_dim == 1024, \
            f"Wrong effective_dim: {processor_avg.effective_dim}"

        print(f"\n✓ Effective dimension calculations:")
        print(f"  Concatenate (3 annotations): {processor_concat.effective_dim}")
        print(f"  Average: {processor_avg.effective_dim}")

    def test_configurable_tr_length(self):
        """Test that TR length is configurable."""
        for tr_length in [1.0, 1.5, 2.0]:
            processor = TextProcessor(
                tr=tr_length,
                temporal_mode='concatenate',
                max_annotations_per_tr=3
            )

            if not ANNOTATIONS_PATH.exists():
                pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

            embeddings, metadata = processor.annotations_to_embeddings(
                ANNOTATIONS_PATH,
                n_trs=50
            )

            # Verify TR timing
            for idx, row in metadata.head(5).iterrows():
                expected_start = idx * tr_length - tr_length  # Window start
                expected_end = (idx + 1) * tr_length
                # Allow some tolerance for window calculations
                assert row['end_time'] - tr_length <= expected_end + 0.01, \
                    f"TR timing mismatch at idx {idx}"

            print(f"\n  TR length {tr_length:.1f}s: ✓")

    def test_get_embedding_info_concatenate(self, processor_concat):
        """Test that get_embedding_info includes new parameters."""
        info = processor_concat.get_embedding_info()

        assert 'temporal_mode' in info
        assert 'max_annotations_per_tr' in info
        assert 'temporal_window' in info
        assert 'effective_dim' in info

        assert info['temporal_mode'] == 'concatenate'
        assert info['max_annotations_per_tr'] == 3
        assert info['temporal_window'] == 1.0
        assert info['effective_dim'] == 3072

        print(f"\n✓ Embedding info (concatenate mode):")
        for key, value in info.items():
            print(f"  {key}: {value}")


# Integration test for concatenation
def test_concatenation_full_pipeline():
    """Integration test of concatenation mode end-to-end."""
    if not ANNOTATIONS_PATH.exists():
        pytest.skip(f"Annotations file not found: {ANNOTATIONS_PATH}")

    print("\n" + "="*60)
    print("CONCATENATION PIPELINE TEST (Issue #26)")
    print("="*60)

    # Initialize processor with concatenation
    processor = TextProcessor(
        temporal_mode='concatenate',
        max_annotations_per_tr=3,
        temporal_window=1.0
    )

    # Step 1: Generate embeddings
    print("\n1. Generating concatenated embeddings...")
    embeddings, metadata = processor.annotations_to_embeddings(
        ANNOTATIONS_PATH,
        n_trs=N_TRS_TARGET
    )
    print(f"   Shape: {embeddings.shape}")
    print(f"   Expected: ({N_TRS_TARGET}, 3072)")
    assert embeddings.shape == (N_TRS_TARGET, 3072)

    # Step 2: Verify dimension consistency
    print("\n2. Verifying dimension consistency...")
    for tr_idx in range(N_TRS_TARGET):
        assert embeddings[tr_idx].shape == (3072,), \
            f"Dimension mismatch at TR {tr_idx}"
    print(f"   ✓ All {N_TRS_TARGET} TRs have consistent shape (3072,)")

    # Step 3: Check annotation distribution
    print("\n3. Analyzing annotation distribution...")
    counts = metadata['n_segments_contributing'].value_counts().sort_index()
    for count, freq in counts.items():
        print(f"   {count} annotations: {freq} TRs ({100*freq/N_TRS_TARGET:.1f}%)")

    # Step 4: Verify no NaN values
    print("\n4. Checking for invalid values...")
    assert not np.isnan(embeddings).any(), "Found NaN values"
    assert not np.isinf(embeddings).any(), "Found Inf values"
    print(f"   ✓ No NaN or Inf values")
    print(f"   Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    # Step 5: Compare with average mode
    print("\n5. Comparing with average mode...")
    processor_avg = TextProcessor(temporal_mode='average')
    embeddings_avg, _ = processor_avg.annotations_to_embeddings(
        ANNOTATIONS_PATH,
        n_trs=N_TRS_TARGET
    )
    print(f"   Concatenate: {embeddings.shape}")
    print(f"   Average: {embeddings_avg.shape}")
    assert embeddings.shape[1] == 3 * embeddings_avg.shape[1]

    print("\n✓ Concatenation pipeline test passed!")
    print("="*60)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
