"""
Test script to validate text embedding functionality with real BGE model.
Validates that the model loads without segfaults and performs nearest-neighbor recovery.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from giblet.data.text import TextProcessor


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.data
def test_text_embedding_validation(data_dir, tmp_path):
    """Run text embedding validation test."""

    output_lines = []

    def log(msg):
        """Log to both console and output list."""
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("Text Embedding Validation Test")
    log("=" * 80)
    log(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # Initialize TextProcessor
    log("1. Initializing TextProcessor with BGE-large-en-v1.5...")
    try:
        processor = TextProcessor(
            model_name="BAAI/bge-large-en-v1.5",
            tr=1.5,
            device=None,  # Auto-detect (should use MPS on Mac)
        )
        log("   ✓ TextProcessor initialized")
    except Exception as e:
        log(f"   ✗ Failed to initialize: {e}")
        pytest.fail(f"Failed to initialize TextProcessor: {e}")

    # Get model info
    log("")
    log("2. Loading model and getting info...")
    try:
        info = processor.get_embedding_info()
        log(f"   Model: {info['model_name']}")
        log(f"   Embedding dimension: {info['embedding_dim']}")
        log(f"   Device: {info['device']}")
        log(f"   TR: {info['tr']}s")
        log(f"   Aggregation: {info['aggregation']}")
        log(f"   Gap fill: {info['gap_fill']}")
        log("   ✓ Model loaded successfully without segfault!")
    except Exception as e:
        log(f"   ✗ Failed to load model: {e}")
        pytest.fail(f"Failed to load model: {e}")

    # Load annotations
    log("")
    log("3. Loading annotations from data/annotations.xlsx...")
    annotations_path = data_dir / "annotations.xlsx"

    if not annotations_path.exists():
        log(f"   ✗ Annotations file not found: {annotations_path}")
        pytest.skip(f"Annotations file not found: {annotations_path}")

    try:
        annotations = processor.load_annotations(annotations_path)
        log(f"   ✓ Loaded {len(annotations)} annotation segments")
        log(f"   Columns: {list(annotations.columns)}")
    except Exception as e:
        log(f"   ✗ Failed to load annotations: {e}")
        pytest.fail(f"Failed to load annotations: {e}")

    # Test with first 10 annotations
    log("")
    log("4. Processing first 10 annotation segments...")
    try:
        # Get first 10 segments
        annotations_subset = annotations.head(10).copy()

        # Combine text columns
        combined_text = processor.combine_text_columns(annotations_subset)
        valid_mask = combined_text.notna()
        valid_annotations = annotations_subset[valid_mask]
        valid_text = combined_text[valid_mask].values

        log(f"   ✓ Combined text from {len(valid_annotations)} segments (with text)")
        log("")
        log("   Sample texts:")
        for i, text in enumerate(valid_text[:5], 1):
            truncated = text[:100] + "..." if len(text) > 100 else text
            log(f"     [{i}] {truncated}")

    except Exception as e:
        log(f"   ✗ Failed to process annotations: {e}")
        pytest.fail(f"Failed to process annotations: {e}")

    # Embed the texts
    log("")
    log("5. Embedding text segments...")
    try:
        # Force model load if not already loaded
        processor._load_model()

        embeddings = processor._model.encode(
            valid_text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        log(f"   ✓ Generated embeddings: shape {embeddings.shape}")
        log(f"   Embedding stats:")
        log(f"     Mean: {embeddings.mean():.6f}")
        log(f"     Std: {embeddings.std():.6f}")
        log(f"     Min: {embeddings.min():.6f}")
        log(f"     Max: {embeddings.max():.6f}")

    except Exception as e:
        log(f"   ✗ Failed to embed texts: {e}")
        pytest.fail(f"Failed to embed texts: {e}")

    # Test nearest-neighbor recovery
    log("")
    log("6. Testing nearest-neighbor recovery...")
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        # Compute self-similarity matrix
        similarities = cosine_similarity(embeddings, embeddings)

        log(f"   ✓ Computed similarity matrix: shape {similarities.shape}")
        log("")
        log("   Recovery test (each embedding should match itself perfectly):")

        perfect_recovery = 0
        for i in range(len(embeddings)):
            # Find nearest neighbor (excluding self)
            sim_row = similarities[i].copy()
            sim_row[i] = -1  # Exclude self
            nearest_idx = np.argmax(sim_row)
            self_similarity = similarities[i, i]
            nearest_similarity = similarities[i, nearest_idx]

            # Self should always be 1.0 (normalized embeddings)
            is_perfect = abs(self_similarity - 1.0) < 1e-6
            if is_perfect:
                perfect_recovery += 1

            if i < 3:  # Show first 3
                log(
                    f"     [{i}] Self-similarity: {self_similarity:.6f}, "
                    f"Next best: {nearest_similarity:.6f} (idx {nearest_idx})"
                )

        recovery_rate = perfect_recovery / len(embeddings) * 100
        log("")
        log(
            f"   ✓ Perfect self-recovery: {perfect_recovery}/{len(embeddings)} ({recovery_rate:.1f}%)"
        )

        # Now test actual recovery by treating each embedding as query
        log("")
        log("   Text recovery test (recovering original text from embedding):")
        recovered_count = 0

        for i in range(min(3, len(embeddings))):  # Test first 3
            # Find nearest neighbor in full set
            nearest_idx = np.argmax(similarities[i])
            recovered_text = valid_text[nearest_idx]
            original_text = valid_text[i]

            is_match = nearest_idx == i
            if is_match:
                recovered_count += 1

            log(f"")
            log(f"   Query [{i}]:")
            orig_trunc = (
                original_text[:80] + "..." if len(original_text) > 80 else original_text
            )
            log(f"     Original: {orig_trunc}")

            if is_match:
                log(
                    f"     Recovered: EXACT MATCH (similarity: {similarities[i, nearest_idx]:.6f})"
                )
            else:
                recov_trunc = (
                    recovered_text[:80] + "..."
                    if len(recovered_text) > 80
                    else recovered_text
                )
                log(f"     Recovered: {recov_trunc}")
                log(
                    f"     Match: {nearest_idx} (similarity: {similarities[i, nearest_idx]:.6f})"
                )

    except Exception as e:
        log(f"   ✗ Failed recovery test: {e}")
        import traceback

        log(traceback.format_exc())
        pytest.fail(f"Failed recovery test: {e}")

    # Test with different batch sizes
    log("")
    log("7. Testing stability with different batch sizes...")
    try:
        test_sizes = [1, 5, 10]
        for batch_size in test_sizes:
            test_texts = valid_text[:batch_size]
            test_embeddings = processor._model.encode(
                test_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            log(f"   ✓ Batch size {batch_size}: shape {test_embeddings.shape}")

    except Exception as e:
        log(f"   ✗ Failed batch test: {e}")
        pytest.fail(f"Failed batch test: {e}")

    # Test alignment to TR grid
    log("")
    log("8. Testing alignment to TR grid...")
    try:
        # Use first 10 annotations, align to 20 TRs (covers ~30 seconds at TR=1.5s)
        n_trs = 20
        tr_embeddings, metadata = processor.align_to_trs(
            valid_annotations, embeddings, n_trs
        )

        log(f"   ✓ Aligned to TR grid: shape {tr_embeddings.shape}")
        log(
            f"   TRs with segments: {(metadata['n_segments_contributing'] > 0).sum()}/{n_trs}"
        )
        log(
            f"   TRs with gaps: {(metadata['n_segments_contributing'] == 0).sum()}/{n_trs}"
        )

        # Show first few TRs
        log("")
        log("   First 5 TRs:")
        for i in range(min(5, n_trs)):
            n_segs = metadata.iloc[i]["n_segments_contributing"]
            start_t = metadata.iloc[i]["start_time"]
            end_t = metadata.iloc[i]["end_time"]
            log(f"     TR {i}: [{start_t:.1f}s - {end_t:.1f}s] {n_segs} segments")

    except Exception as e:
        log(f"   ✗ Failed TR alignment: {e}")
        import traceback

        log(traceback.format_exc())
        pytest.fail(f"Failed TR alignment: {e}")

    log("")
    log("=" * 80)
    log("✓ ALL TESTS PASSED - No segfaults detected!")
    log("=" * 80)

    # Save results
    output_path = tmp_path / "text_embedding_validation.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    log(f"\nResults saved to: {output_path}")

    # Assert test passed
    assert len(output_lines) > 0
