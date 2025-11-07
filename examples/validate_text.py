"""
Text modality validation script.

Performs comprehensive round-trip validation:
1. Load real annotations.xlsx (1000 segments)
2. Convert to text embeddings aligned to TRs
3. Test nearest-neighbor recovery (top-1, top-5, top-10)
4. Calculate semantic similarity (cosine similarity)
5. Test recovery accuracy
6. Save results to text file for manual review
7. Verify timing alignment with Issue #10 fix

All tests use REAL data - NO MOCKS.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.data.text import TextProcessor


def calculate_recovery_accuracy(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    processor: TextProcessor,
    k_values: list = [1, 5, 10],
) -> dict:
    """
    Calculate top-k recovery accuracy.

    For each TR, check if the original text that contributed to it
    appears in the top-k nearest neighbors.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_trs, embedding_dim) TR embeddings
    metadata : pd.DataFrame
        Metadata with segment information
    processor : TextProcessor
        TextProcessor instance
    k_values : list
        List of k values to test

    Returns
    -------
    accuracy : dict
        Dictionary with top-k accuracy for each k
    """
    # Get segment embeddings and texts
    segment_embeddings = metadata["segment_embeddings"].iloc[0]
    segment_texts = metadata["segment_texts"].iloc[0]

    # Calculate similarities between all TRs and all segments
    similarities = cosine_similarity(embeddings, segment_embeddings)

    accuracies = {}

    for k in k_values:
        correct = 0
        total = 0

        for tr_idx in range(len(embeddings)):
            # Get contributing segments for this TR
            contributing = metadata.iloc[tr_idx]["segment_indices"]

            if len(contributing) == 0:
                # Skip TRs with no contributing segments (gaps)
                continue

            # Get top-k predictions
            top_k_indices = np.argsort(similarities[tr_idx])[-k:][::-1]

            # Check if any contributing segment is in top-k
            if any(seg_idx in top_k_indices for seg_idx in contributing):
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0
        accuracies[f"top_{k}"] = accuracy

    return accuracies


def test_semantic_consistency(embeddings: np.ndarray, metadata: pd.DataFrame) -> dict:
    """
    Test semantic consistency between consecutive TRs.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_trs, embedding_dim) TR embeddings
    metadata : pd.DataFrame
        Metadata with segment information

    Returns
    -------
    stats : dict
        Statistics about semantic consistency
    """
    n_trs = len(embeddings)

    # Calculate cosine similarities between consecutive TRs
    consecutive_sims = []
    for i in range(n_trs - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        consecutive_sims.append(sim)

    consecutive_sims = np.array(consecutive_sims)

    # Calculate similarities between random pairs
    n_random = 1000
    random_sims = []
    for _ in range(n_random):
        i = np.random.randint(0, n_trs)
        j = np.random.randint(0, n_trs)
        if i != j:
            sim = np.dot(embeddings[i], embeddings[j])
            random_sims.append(sim)

    random_sims = np.array(random_sims)

    return {
        "consecutive_mean": np.mean(consecutive_sims),
        "consecutive_std": np.std(consecutive_sims),
        "consecutive_min": np.min(consecutive_sims),
        "consecutive_max": np.max(consecutive_sims),
        "random_mean": np.mean(random_sims),
        "random_std": np.std(random_sims),
        "random_min": np.min(random_sims),
        "random_max": np.max(random_sims),
    }


def analyze_gap_filling(metadata: pd.DataFrame) -> dict:
    """
    Analyze gap filling behavior.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with n_segments_contributing

    Returns
    -------
    stats : dict
        Gap statistics
    """
    n_trs = len(metadata)
    gaps = (metadata["n_segments_contributing"] == 0).sum()
    coverage = (metadata["n_segments_contributing"] > 0).sum()
    multi_contributor = (metadata["n_segments_contributing"] > 1).sum()

    return {
        "total_trs": n_trs,
        "trs_with_segments": coverage,
        "trs_with_gaps": gaps,
        "trs_with_multiple_segments": multi_contributor,
        "coverage_pct": 100 * coverage / n_trs,
        "gap_pct": 100 * gaps / n_trs,
        "multi_contributor_pct": 100 * multi_contributor / n_trs,
    }


def save_recovery_examples(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    processor: TextProcessor,
    output_path: Path,
    n_examples: int = 20,
):
    """
    Save examples of text recovery for manual inspection.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_trs, embedding_dim) TR embeddings
    metadata : pd.DataFrame
        Metadata with segment information
    processor : TextProcessor
        TextProcessor instance
    output_path : Path
        Path to save examples
    n_examples : int
        Number of examples to save
    """
    # Get segment embeddings and texts
    segment_embeddings = metadata["segment_embeddings"].iloc[0]
    segment_texts = metadata["segment_texts"].iloc[0]

    # Calculate similarities
    similarities = cosine_similarity(embeddings, segment_embeddings)

    # Select TRs to showcase
    # Include beginning, middle, end, and some with multiple contributors
    example_trs = []

    # Add fixed TRs
    example_trs.extend([0, 100, 300, 500, 700, 900])

    # Add TRs with multiple contributors
    multi_contributor_trs = metadata[metadata["n_segments_contributing"] > 1][
        "tr_index"
    ].values
    if len(multi_contributor_trs) > 0:
        example_trs.extend(
            np.random.choice(
                multi_contributor_trs,
                size=min(5, len(multi_contributor_trs)),
                replace=False,
            )
        )

    # Add random TRs
    random_trs = np.random.choice(
        len(embeddings),
        size=min(n_examples - len(example_trs), len(embeddings)),
        replace=False,
    )
    example_trs.extend(random_trs)

    example_trs = sorted(list(set(example_trs)))[:n_examples]

    # Write to file
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TEXT RECOVERY VALIDATION EXAMPLES\n")
        f.write("=" * 80 + "\n\n")

        for tr_idx in example_trs:
            f.write(f"\n{'='*80}\n")
            f.write(f"TR {tr_idx} (Time: {tr_idx * 1.5:.1f}s)\n")
            f.write(f"{'='*80}\n\n")

            # Get metadata
            tr_meta = metadata.iloc[tr_idx]
            n_contributing = tr_meta["n_segments_contributing"]
            contributing_indices = tr_meta["segment_indices"]

            f.write(f"Contributing segments: {n_contributing}\n")

            if n_contributing > 0:
                f.write(f"\nOriginal text(s) for this TR:\n")
                for i, seg_idx in enumerate(contributing_indices, 1):
                    f.write(f"  {i}. {segment_texts[seg_idx]}\n")
            else:
                f.write(f"\n(No original segments - this TR was gap-filled)\n")

            # Get top-5 predictions
            top_5_indices = np.argsort(similarities[tr_idx])[-5:][::-1]
            top_5_sims = similarities[tr_idx][top_5_indices]

            f.write(f"\nTop-5 recovered texts:\n")
            for i, (seg_idx, sim) in enumerate(zip(top_5_indices, top_5_sims), 1):
                # Mark if this was an original contributor
                marker = " ✓" if seg_idx in contributing_indices else ""
                f.write(f"  {i}. [Similarity: {sim:.4f}]{marker}\n")
                f.write(f"     {segment_texts[seg_idx]}\n")

            f.write("\n")

    print(f"  Saved recovery examples to {output_path.name}")


def validate_text_modality():
    """
    Main validation function for text modality.
    """
    print("\n" + "=" * 80)
    print("TEXT MODALITY VALIDATION")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    annotations_path = project_root / "data" / "annotations.xlsx"
    output_dir = project_root / "validation_outputs"
    output_dir.mkdir(exist_ok=True)

    if not annotations_path.exists():
        print(f"\n❌ ERROR: Annotations file not found: {annotations_path}")
        print("Please ensure annotations.xlsx is in the data directory")
        return

    print(f"\nAnnotations path: {annotations_path}")
    print(f"Output directory: {output_dir}")

    # Initialize processor
    processor = TextProcessor(
        model_name="BAAI/bge-large-en-v1.5",
        tr=1.5,
        aggregation="mean",
        gap_fill="forward_fill",
    )

    # Get model info
    print("\n" + "=" * 80)
    print("PROCESSOR INFORMATION")
    print("=" * 80)
    info = processor.get_embedding_info()
    print(f"\nEmbedding model:")
    print(f"  Model: {info['model_name']}")
    print(f"  Embedding dimension: {info['embedding_dim']}")
    print(f"  Device: {info['device']}")
    print(f"  TR: {info['tr']}s")
    print(f"  Aggregation: {info['aggregation']}")
    print(f"  Gap fill: {info['gap_fill']}")

    # Load annotations
    print("\n" + "=" * 80)
    print("LOADING ANNOTATIONS")
    print("=" * 80)
    annotations = processor.load_annotations(annotations_path)
    print(f"\nLoaded {len(annotations)} annotation segments")
    print(
        f"Time range: {annotations['Start Time (s)'].min():.1f}s to "
        f"{annotations['End Time (s)'].max():.1f}s"
    )
    print(f"Duration: {annotations['End Time (s)'].max():.1f}s")

    # Show sample annotations
    print(f"\nSample annotations:")
    for i in [0, 100, 500, 999]:
        row = annotations.iloc[i]
        print(f"  Segment {i}: {row['Start Time (s)']:.1f}s-{row['End Time (s)']:.1f}s")
        if "Scene Details - A Level" in annotations.columns:
            details = row["Scene Details - A Level"]
            if pd.notna(details):
                print(f"    {str(details)[:70]}...")

    # Convert to embeddings
    print("\n" + "=" * 80)
    print("GENERATING EMBEDDINGS")
    print("=" * 80)

    n_trs = 950  # ~23 minutes at TR=1.5s
    print(f"\nTarget TRs: {n_trs} (duration: {n_trs * 1.5:.1f}s)")

    embeddings, metadata = processor.annotations_to_embeddings(
        annotations_path, n_trs=n_trs
    )

    print(f"\n✓ Generated embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"  Mean: {embeddings.mean():.3f}, Std: {embeddings.std():.3f}")

    # Analyze gap filling
    print("\n" + "=" * 80)
    print("TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 80)

    gap_stats = analyze_gap_filling(metadata)
    print(f"\nCoverage statistics:")
    print(f"  Total TRs: {gap_stats['total_trs']}")
    print(
        f"  TRs with segments: {gap_stats['trs_with_segments']} ({gap_stats['coverage_pct']:.1f}%)"
    )
    print(
        f"  TRs with gaps: {gap_stats['trs_with_gaps']} ({gap_stats['gap_pct']:.1f}%)"
    )
    print(
        f"  TRs with multiple segments: {gap_stats['trs_with_multiple_segments']} "
        f"({gap_stats['multi_contributor_pct']:.1f}%)"
    )

    # Find gap regions
    gap_mask = metadata["n_segments_contributing"] == 0
    if gap_mask.any():
        gap_indices = metadata[gap_mask]["tr_index"].values
        print(f"\nGap regions (first 20):")
        gap_ranges = []
        start = gap_indices[0]
        prev = gap_indices[0]

        for idx in gap_indices[1:]:
            if idx != prev + 1:
                gap_ranges.append((start, prev))
                start = idx
            prev = idx
        gap_ranges.append((start, prev))

        for start, end in gap_ranges[:20]:
            print(f"  TRs {start}-{end} ({start*1.5:.1f}s-{end*1.5:.1f}s)")

    # Test semantic consistency
    print("\n" + "=" * 80)
    print("SEMANTIC CONSISTENCY TEST")
    print("=" * 80)

    consistency_stats = test_semantic_consistency(embeddings, metadata)
    print(f"\nConsecutive TR similarities:")
    print(f"  Mean: {consistency_stats['consecutive_mean']:.4f}")
    print(f"  Std: {consistency_stats['consecutive_std']:.4f}")
    print(
        f"  Range: [{consistency_stats['consecutive_min']:.4f}, "
        f"{consistency_stats['consecutive_max']:.4f}]"
    )

    print(f"\nRandom TR pair similarities:")
    print(f"  Mean: {consistency_stats['random_mean']:.4f}")
    print(f"  Std: {consistency_stats['random_std']:.4f}")
    print(
        f"  Range: [{consistency_stats['random_min']:.4f}, "
        f"{consistency_stats['random_max']:.4f}]"
    )

    # Check if consecutive TRs are more similar than random pairs
    if consistency_stats["consecutive_mean"] > consistency_stats["random_mean"]:
        diff = consistency_stats["consecutive_mean"] - consistency_stats["random_mean"]
        print(f"\n✓ Consecutive TRs are more similar than random pairs (+{diff:.4f})")
        print(f"  This indicates temporal coherence in the embeddings")
    else:
        print(f"\n⚠ WARNING: Consecutive TRs are not more similar than random pairs")

    # Test recovery accuracy
    print("\n" + "=" * 80)
    print("RECOVERY ACCURACY TEST")
    print("=" * 80)

    print(f"\nTesting nearest-neighbor recovery...")
    accuracies = calculate_recovery_accuracy(
        embeddings, metadata, processor, k_values=[1, 5, 10, 20]
    )

    print(f"\nRecovery accuracy:")
    for k, acc in accuracies.items():
        k_val = int(k.split("_")[1])
        print(f"  Top-{k_val:2d}: {acc:6.2%}")

    # Test with actual recovery
    print("\n" + "=" * 80)
    print("TEXT RECOVERY TEST")
    print("=" * 80)

    print(f"\nRecovering texts for all TRs...")
    recovered_texts_top1 = processor.embeddings_to_text(
        embeddings, metadata, method="nearest_neighbor", top_k=1
    )
    recovered_texts_top5 = processor.embeddings_to_text(
        embeddings, metadata, method="nearest_neighbor", top_k=5
    )

    print(f"✓ Recovered {len(recovered_texts_top1)} texts (top-1)")
    print(f"✓ Recovered {len(recovered_texts_top5)} texts (top-5)")

    # Show some examples
    print(f"\nSample recovered texts (top-1):")
    for tr_idx in [0, 100, 300, 500, 700, 900]:
        text = recovered_texts_top1[tr_idx]
        print(f"  TR {tr_idx:3d} ({tr_idx*1.5:6.1f}s): {text[:70]}...")

    # Save detailed examples
    print("\n" + "=" * 80)
    print("SAVING RECOVERY EXAMPLES")
    print("=" * 80)

    examples_path = output_dir / "text_recovery_examples.txt"
    save_recovery_examples(
        embeddings, metadata, processor, examples_path, n_examples=30
    )

    # Calculate semantic similarities
    print("\n" + "=" * 80)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 80)

    # Get segment embeddings
    segment_embeddings = metadata["segment_embeddings"].iloc[0]
    segment_texts = metadata["segment_texts"].iloc[0]

    # Calculate pairwise similarities between segments
    segment_sims = cosine_similarity(segment_embeddings)

    print(f"\nSegment-to-segment similarities:")
    print(f"  Mean: {np.mean(segment_sims):.4f}")
    print(f"  Std: {np.std(segment_sims):.4f}")
    print(f"  Min: {np.min(segment_sims):.4f}")
    print(f"  Max: {np.max(segment_sims):.4f}")

    # Find most similar segment pairs
    print(f"\nTop-5 most similar segment pairs:")
    n_segments = len(segment_texts)
    for i in range(5):
        # Get highest similarity (excluding diagonal)
        sim_copy = segment_sims.copy()
        np.fill_diagonal(sim_copy, -1)
        max_idx = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)
        max_sim = segment_sims[max_idx]

        print(f"\n  {i+1}. Similarity: {max_sim:.4f}")
        print(f"     Segment {max_idx[0]}: {segment_texts[max_idx[0]][:60]}...")
        print(f"     Segment {max_idx[1]}: {segment_texts[max_idx[1]][:60]}...")

        # Zero out to find next
        segment_sims[max_idx] = -1
        segment_sims[max_idx[1], max_idx[0]] = -1

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nAnnotations: {len(annotations)} segments")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Coverage: {gap_stats['coverage_pct']:.1f}%")
    print(f"Recovery accuracy (top-1): {accuracies['top_1']:.2%}")
    print(f"Recovery accuracy (top-5): {accuracies['top_5']:.2%}")
    print(f"Recovery accuracy (top-10): {accuracies['top_10']:.2%}")
    print(f"Consecutive similarity: {consistency_stats['consecutive_mean']:.4f}")

    print(f"\nGenerated outputs:")
    print(f"  {examples_path.name}")

    # Final verdict
    print("\n" + "=" * 80)
    if (
        accuracies["top_1"] > 0.5
        and accuracies["top_5"] > 0.8
        and gap_stats["coverage_pct"] > 70
    ):
        print("✅ TEXT VALIDATION PASSED - GOOD QUALITY")
    elif (
        accuracies["top_1"] > 0.3
        and accuracies["top_5"] > 0.6
        and gap_stats["coverage_pct"] > 50
    ):
        print("✅ TEXT VALIDATION PASSED - ACCEPTABLE QUALITY")
    else:
        print("⚠ TEXT VALIDATION WARNING - QUALITY BELOW EXPECTED")
    print("=" * 80)

    return {
        "n_segments": len(annotations),
        "n_trs": n_trs,
        "coverage_pct": gap_stats["coverage_pct"],
        "accuracies": accuracies,
        "consistency": consistency_stats,
        "gap_stats": gap_stats,
    }


if __name__ == "__main__":
    try:
        results = validate_text_modality()
        print(f"\n✓ Text validation complete.")
        print(f"  Top-1 accuracy: {results['accuracies']['top_1']:.2%}")
        print(f"  Top-5 accuracy: {results['accuracies']['top_5']:.2%}")
        print(f"  Coverage: {results['coverage_pct']:.1f}%")
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
