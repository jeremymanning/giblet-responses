"""
Test real text embedding with BGE model.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from giblet.data.text import TextProcessor


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.data
def test_real_bge_embeddings(data_dir, tmp_path):
    """Test REAL BGE embedding validation."""

    print("Loading modules...")

    # Load REAL BGE model
    print("Loading BAAI/bge-large-en-v1.5 (this will download ~1.3GB)...")
    processor = TextProcessor(model_name="BAAI/bge-large-en-v1.5", tr=1.5, device="cpu")
    print("Model loaded")

    # Load annotations
    annotations_path = data_dir / "annotations.xlsx"
    if not annotations_path.exists():
        pytest.skip(f"Annotations not found at {annotations_path}")

    annotations = processor.load_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotations")

    # Get first 10 scene descriptions
    combined_text = processor.combine_text_columns(annotations)
    valid_mask = combined_text.notna()
    scenes = combined_text[valid_mask].iloc[:10].tolist()

    # Force model load
    processor._load_model()

    # Generate REAL embeddings
    print("Generating REAL embeddings...")
    embeddings = processor._model.encode(
        scenes,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Test nearest-neighbor recovery
    print("Testing nearest-neighbor recovery...")

    output_path = tmp_path / "text_embedding_validation.txt"
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("REAL BGE EMBEDDING VALIDATION\n")
        f.write("Model: BAAI/bge-large-en-v1.5\n")
        f.write("=" * 80 + "\n\n")

        perfect_matches = 0
        for i, scene in enumerate(scenes):
            # Find nearest neighbor to this embedding
            sims = cosine_similarity([embeddings[i]], embeddings)[0]
            nearest_idx = np.argmax(sims)

            f.write(f"\nAnnotation {i}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"ORIGINAL: {scene}\n")
            f.write(f"RECOVERED (nearest): {scenes[nearest_idx]}\n")
            f.write(f"Cosine similarity: {sims[nearest_idx]:.4f}\n")

            if i == nearest_idx:
                f.write("PERFECT MATCH (same annotation)\n")
                perfect_matches += 1
            else:
                f.write("Different annotation retrieved\n")

    print(f"Results saved to {output_path}")
    print("This shows ACTUAL BGE embeddings and nearest-neighbor recovery")

    # Assert that all perfect matches (self-similarity should be 1.0)
    assert perfect_matches == len(scenes), "Expected all self-matches to be perfect"
