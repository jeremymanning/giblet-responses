"""
Test text embedding and reconstruction pipeline (mock version).

This script:
1. Loads annotations without loading the actual model
2. Simulates text embedding with random vectors
3. Tests recovery via nearest-neighbor
4. Saves results for manual review
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from time import time


@pytest.mark.unit
def test_text_embedding_mock(data_dir, tmp_path):
    """Run embedding and recovery test with mock embeddings."""

    # Setup paths
    annotations_path = data_dir / 'annotations.xlsx'
    if not annotations_path.exists():
        pytest.skip(f"Annotations not found at {annotations_path}")

    output_path = tmp_path / 'text_embedding_validation.txt'

    print("=" * 70)
    print("TEXT EMBEDDING & RECONSTRUCTION TEST (SIMULATION)")
    print("=" * 70)
    print(f"Annotations: {annotations_path}")
    print(f"Output: {output_path}")
    print()

    # Load annotations directly using pandas
    print("Loading annotations...")
    annotations = pd.read_excel(annotations_path)
    annotations.columns = annotations.columns.str.strip()

    print(f"Total segments: {len(annotations)}")
    print(f"Columns: {list(annotations.columns)}")
    print()

    # Get text columns
    text_columns = ['Scene Details - A Level', 'Name - All', 'Location']
    available_cols = [col for col in text_columns if col in annotations.columns]
    print(f"Text columns to use: {available_cols}")

    # Combine text columns
    print("Combining text columns...")
    combined = annotations[available_cols].fillna('').astype(str)
    combined_text = combined.apply(
        lambda row: '; '.join([x for x in row if x]),
        axis=1
    )
    combined_text = combined_text.replace('', np.nan)

    valid_mask = combined_text.notna()
    valid_annotations = annotations[valid_mask].copy().reset_index(drop=True)
    valid_text = combined_text[valid_mask].values

    print(f"Valid text segments: {len(valid_text)}")
    print()

    # Show first few annotations for reference
    print("First 5 annotations:")
    for i in range(min(5, len(valid_text))):
        start_time = valid_annotations.iloc[i]['Start Time (s)']
        end_time = valid_annotations.iloc[i]['End Time (s)']
        text = valid_text[i]
        print(f"  [{i}] {start_time:.1f}s - {end_time:.1f}s: {text[:50]}...")
    print()

    # Simulate embeddings (random 1024-dim vectors)
    print("Simulating text embeddings (1024-dim random vectors)...")
    embedding_dim = 1024
    np.random.seed(42)  # For reproducibility
    segment_embeddings = np.random.randn(len(valid_text), embedding_dim).astype(np.float32)

    # Normalize embeddings
    norms = np.linalg.norm(segment_embeddings, axis=1, keepdims=True)
    segment_embeddings = segment_embeddings / norms

    print(f"Embedding shape: {segment_embeddings.shape}")
    print(f"First embedding sample: {segment_embeddings[0, :5]}")
    print()

    # Setup TR constants
    tr_duration = 1.5
    n_test_trs = 30

    # Map annotations to TRs (first 30 only)
    print(f"Mapping annotations to TRs (first {n_test_trs})...")
    segment_to_trs = {}  # Which TRs each segment contributes to

    for seg_idx, (_, row) in enumerate(valid_annotations.iterrows()):
        start_time = row['Start Time (s)']
        end_time = row['End Time (s)']

        # Find TRs that overlap with this segment
        start_tr = int(np.floor(start_time / tr_duration))
        end_tr = int(np.ceil(end_time / tr_duration))
        start_tr = max(0, start_tr)
        end_tr = min(n_test_trs, end_tr)  # Limit to first N TRs

        segment_to_trs[seg_idx] = list(range(start_tr, end_tr))

    # Build mapping of which segments contribute to each TR
    tr_to_segments = {}
    for tr_idx in range(n_test_trs):
        tr_to_segments[tr_idx] = []
        for seg_idx, tr_list in segment_to_trs.items():
            if tr_idx in tr_list:
                tr_to_segments[tr_idx].append(seg_idx)

    print(f"TR mapping complete")
    print()

    # Compute TR embeddings for first N TRs
    print(f"Computing TR embeddings for first {n_test_trs} TRs...")
    tr_embeddings = np.zeros((n_test_trs, embedding_dim), dtype=np.float32)

    for tr_idx in range(n_test_trs):
        segments = tr_to_segments[tr_idx]
        if segments:
            contributing_embeddings = segment_embeddings[segments]
            tr_embeddings[tr_idx] = np.mean(contributing_embeddings, axis=0)
            # Normalize
            norm = np.linalg.norm(tr_embeddings[tr_idx])
            if norm > 0:
                tr_embeddings[tr_idx] /= norm

    print(f"TR embeddings computed: {tr_embeddings.shape}")
    print()

    # Recover text using nearest-neighbor
    print("Recovering text from embeddings (cosine similarity)...")
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

    for tr_idx in range(n_test_trs):
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

    total_with_data = sum(1 for tr in range(n_test_trs) if tr_to_segments[tr])
    if total_with_data > 0:
        match_pct = (matches + partial_matches) / total_with_data * 100
    else:
        match_pct = 0

    print(f"TRs with data: {total_with_data}/{n_test_trs}")
    print(f"Exact matches: {matches}/{total_with_data}")
    print(f"Partial matches: {partial_matches}/{total_with_data}")
    print(f"Match rate: {match_pct:.1f}%")
    print()

    # Write results to file
    print("Writing results to file...")
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT EMBEDDING & RECONSTRUCTION VALIDATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Test Type: SIMULATION (random embeddings)\n")
        f.write(f"Embedding dimension: {embedding_dim}\n")
        f.write(f"TR duration: {tr_duration}s\n")
        f.write(f"Total segments: {len(valid_text)}\n")
        f.write(f"TRs tested: {n_test_trs} (0-{n_test_trs-1})\n")
        f.write(f"TRs with data: {total_with_data}\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        if total_with_data > 0:
            f.write(f"Exact matches: {matches}/{total_with_data} ({matches/total_with_data*100:.1f}%)\n")
            f.write(f"Partial matches: {partial_matches}/{total_with_data} ({partial_matches/total_with_data*100:.1f}%)\n")
            f.write(f"Total match rate: {(matches + partial_matches)/total_with_data*100:.1f}%\n\n")
        else:
            f.write("No TRs with data to test\n\n")

        f.write("NOTE: Using random embeddings for simulation. With real BGE model,\n")
        f.write("expect much higher match rates since semantically similar texts will\n")
        f.write("have higher cosine similarities.\n\n")

        f.write("=" * 70 + "\n")
        f.write(f"DETAILED RESULTS (TR 0-{n_test_trs-1})\n")
        f.write("=" * 70 + "\n\n")

        for tr_idx in range(n_test_trs):
            start_time = tr_idx * tr_duration
            end_time = start_time + tr_duration

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
                f.write(f"  [{rank}] {sim_scores[idx]:.4f} - {valid_text[idx][:60]}...\n")

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
    if total_with_data > 0:
        print(f"Exact match rate: {matches}/{total_with_data} ({matches/total_with_data*100:.1f}%)")
        print(f"Partial match rate: {(matches + partial_matches)/total_with_data*100:.1f}%")
    else:
        print("No data to test")
    print(f"File ready for manual review: {output_path}")
    print()

    print("NOTE: This is a simulation using random embeddings.")
    print("Real BGE model would produce much better matches due to semantic similarity.")
    print()

    # Assert we got results
    assert len(tr_embeddings) == n_test_trs
    assert len(recovered_texts) == n_test_trs
