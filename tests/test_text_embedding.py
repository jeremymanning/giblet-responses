"""
Test text embedding and reconstruction pipeline.

This script:
1. Loads TextProcessor
2. Processes first 30 TRs from annotations
3. Embeds text using BGE model
4. Recovers text via nearest-neighbor
5. Saves results for manual review
"""

from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pytest

from giblet.data.text import TextProcessor


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.data
def test_text_embedding_reconstruction(data_dir, tmp_path):
    """Run embedding and recovery test."""

    # Setup paths
    annotations_path = data_dir / "annotations.xlsx"
    if not annotations_path.exists():
        pytest.skip(f"Annotations not found at {annotations_path}")

    output_path = tmp_path / "text_embedding_validation.txt"

    print("=" * 70)
    print("TEXT EMBEDDING & RECONSTRUCTION TEST")
    print("=" * 70)
    print(f"Annotations: {annotations_path}")
    print(f"Output: {output_path}")
    print()

    # Initialize TextProcessor
    print("Initializing TextProcessor...")
    processor = TextProcessor(
        model_name="BAAI/bge-large-en-v1.5",
        tr=1.5,
        aggregation="mean",
        gap_fill="forward_fill",
    )
    print(f"Model config: {processor.get_embedding_info()}")
    print()

    # Load annotations
    print("Loading annotations...")
    annotations = processor.load_annotations(annotations_path)
    print(f"Total segments: {len(annotations)}")
    print(f"Columns: {list(annotations.columns)}")
    print()

    # Combine text columns
    print("Combining text columns...")
    combined_text = processor.combine_text_columns(annotations)
    valid_mask = combined_text.notna()
    valid_annotations = annotations[valid_mask].copy()
    valid_text = combined_text[valid_mask].values
    print(f"Valid text segments: {len(valid_text)}")
    print()

    # Show first few annotations for reference
    print("First 5 annotations:")
    for i in range(min(5, len(valid_text))):
        start_time = valid_annotations.iloc[i]["Start Time (s)"]
        end_time = valid_annotations.iloc[i]["End Time (s)"]
        text = valid_text[i]
        print(f"  [{i}] {start_time:.1f}s - {end_time:.1f}s: {text[:50]}...")
    print()

    # Load and embed text
    print("Loading embedding model...")
    processor._load_model()
    print(f"Model device: {processor._model.device}")
    print()

    print("Embedding text segments...")
    start_embed = time()
    segment_embeddings = processor._model.encode(
        valid_text,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embed_time = time() - start_embed
    print(f"Embedded {len(segment_embeddings)} segments in {embed_time:.2f}s")
    print(f"Embedding shape: {segment_embeddings.shape}")
    print()

    # Map annotations to TRs (first 30 only)
    print("Mapping annotations to TRs (first 30)...")
    tr_times = []
    segment_to_trs = {}  # Which TRs each segment contributes to

    for seg_idx, (_, row) in enumerate(valid_annotations.iterrows()):
        start_time = row["Start Time (s)"]
        end_time = row["End Time (s)"]

        # Find TRs that overlap with this segment
        start_tr = int(np.floor(start_time / processor.tr))
        end_tr = int(np.ceil(end_time / processor.tr))
        start_tr = max(0, start_tr)
        end_tr = min(30, end_tr)  # Limit to first 30 TRs

        segment_to_trs[seg_idx] = list(range(start_tr, end_tr))

    # Build mapping of which segments contribute to each TR
    tr_to_segments = {}
    for tr_idx in range(30):
        tr_to_segments[tr_idx] = []
        for seg_idx, tr_list in segment_to_trs.items():
            if tr_idx in tr_list:
                tr_to_segments[tr_idx].append(seg_idx)

    print(f"TR mapping complete")
    print()

    # Compute TR embeddings for first 30 TRs
    print("Computing TR embeddings for first 30 TRs...")
    tr_embeddings = np.zeros((30, processor.n_features), dtype=np.float32)

    for tr_idx in range(30):
        segments = tr_to_segments[tr_idx]
        if segments:
            contributing_embeddings = segment_embeddings[segments]
            tr_embeddings[tr_idx] = np.mean(contributing_embeddings, axis=0)

    print(f"TR embeddings computed: {tr_embeddings.shape}")
    print()

    # Recover text using nearest-neighbor
    print("Recovering text from embeddings...")
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(tr_embeddings, segment_embeddings)
    recovered_indices = np.argmax(similarities, axis=1)
    recovered_texts = [valid_text[idx] for idx in recovered_indices]

    print(f"Text recovery complete")
    print()

    # Compute match statistics
    print("Computing match statistics...")
    matches = 0
    partial_matches = 0

    for tr_idx in range(30):
        segments = tr_to_segments[tr_idx]
        if segments:
            original_texts = [valid_text[seg_idx] for seg_idx in segments]
            recovered = recovered_texts[tr_idx]

            # Check for exact match
            if recovered in original_texts:
                matches += 1
            # Check for partial match (first 50 chars)
            elif any(recovered[:50] == orig[:50] for orig in original_texts):
                partial_matches += 1

    print(f"Exact matches: {matches}/30")
    print(f"Partial matches: {partial_matches}/30")
    print(f"Match rate: {(matches + partial_matches) / 30 * 100:.1f}%")
    print()

    # Write results to file
    print("Writing results to file...")
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT EMBEDDING & RECONSTRUCTION VALIDATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {processor.model_name}\n")
        f.write(f"Embedding dimension: {processor.n_features}\n")
        f.write(f"TR duration: {processor.tr}s\n")
        f.write(f"Aggregation: {processor.aggregation}\n")
        f.write(f"Total segments: {len(valid_text)}\n")
        f.write(f"TRs tested: 30 (0-29)\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Exact matches: {matches}/30 ({matches/30*100:.1f}%)\n")
        f.write(
            f"Partial matches: {partial_matches}/30 ({partial_matches/30*100:.1f}%)\n"
        )
        f.write(f"Total match rate: {(matches + partial_matches)/30*100:.1f}%\n\n")

        f.write("=" * 70 + "\n")
        f.write("DETAILED RESULTS (TR 0-29)\n")
        f.write("=" * 70 + "\n\n")

        for tr_idx in range(30):
            start_time = tr_idx * processor.tr
            end_time = start_time + processor.tr

            f.write(f"TR {tr_idx} [{start_time:.1f}s - {end_time:.1f}s]\n")
            f.write("-" * 70 + "\n")

            segments = tr_to_segments[tr_idx]
            if not segments:
                f.write("ORIGINAL: [No segments contributing]\n")
                f.write("RECOVERED: [No text]\n")
                f.write("STATUS: SKIPPED (no data)\n\n")
                continue

            # Show original text(s)
            original_texts = [valid_text[seg_idx] for seg_idx in segments]
            if len(original_texts) == 1:
                f.write(f"ORIGINAL: {original_texts[0]}\n")
            else:
                f.write(f"ORIGINAL (multiple segments):\n")
                for i, text in enumerate(original_texts):
                    f.write(f"  [{i}] {text}\n")

            # Show recovered text
            recovered = recovered_texts[tr_idx]
            f.write(f"RECOVERED: {recovered}\n")

            # Check match status
            if recovered in original_texts:
                status = "EXACT MATCH"
                matches_found = True
            elif any(recovered[:50] == orig[:50] for orig in original_texts):
                status = "PARTIAL MATCH"
                matches_found = True
            else:
                status = "NO MATCH"
                matches_found = False

            f.write(f"STATUS: {status}\n")

            # Show similarity scores
            sim_scores = similarities[tr_idx]
            top_3_indices = np.argsort(sim_scores)[-3:][::-1]
            f.write(f"TOP 3 SIMILARITIES:\n")
            for rank, idx in enumerate(top_3_indices, 1):
                f.write(
                    f"  [{rank}] {sim_scores[idx]:.4f} - {valid_text[idx][:60]}...\n"
                )

            f.write("\n")

    print(f"Results written to: {output_path}")
    print()

    # Print file size
    file_size = output_path.stat().st_size
    print(f"Output file size: {file_size} bytes")
    print()

    # Display summary
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Exact match rate: {matches}/30 ({matches/30*100:.1f}%)")
    print(f"Partial match rate: {(matches + partial_matches)/30*100:.1f}%")
    print(f"File ready for manual review: {output_path}")
    print()

    # Assert we got some results
    assert len(tr_embeddings) == 30
    assert len(recovered_texts) == 30
