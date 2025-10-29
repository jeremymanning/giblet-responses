import sys
sys.path.insert(0, '.')

print("Loading modules...")
try:
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("✓ All imports successful\n")
    
    # Load REAL BGE model
    print("Loading BAAI/bge-large-en-v1.5 (this will download ~1.3GB)...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print("✓ BGE model loaded\n")
    
    # Load annotations
    df = pd.read_excel('data/annotations.xlsx')
    print(f"✓ Loaded {len(df)} annotations\n")
    
    # Get first 10 scene descriptions
    scenes = df['Scene Details - A Level '].iloc[:10].tolist()
    
    # Generate REAL embeddings
    print("Generating REAL embeddings...")
    embeddings = model.encode(scenes, normalize_embeddings=True)
    print(f"✓ Embeddings shape: {embeddings.shape}\n")
    
    # Test nearest-neighbor recovery
    print("Testing nearest-neighbor recovery...\n")
    
    with open('text_embedding_validation.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("REAL BGE EMBEDDING VALIDATION\n")
        f.write("Model: BAAI/bge-large-en-v1.5\n")
        f.write("="*80 + "\n\n")
        
        for i, scene in enumerate(scenes):
            # Find nearest neighbor to this embedding
            sims = cosine_similarity([embeddings[i]], embeddings)[0]
            nearest_idx = np.argmax(sims)
            
            f.write(f"\nAnnotation {i}:\n")
            f.write("-"*80 + "\n")
            f.write(f"ORIGINAL: {scene}\n")
            f.write(f"RECOVERED (nearest): {scenes[nearest_idx]}\n")
            f.write(f"Cosine similarity: {sims[nearest_idx]:.4f}\n")
            
            if i == nearest_idx:
                f.write("✅ PERFECT MATCH (same annotation)\n")
            else:
                f.write("⚠️  Different annotation retrieved\n")
    
    print("✅ Results saved to text_embedding_validation.txt")
    print("This shows ACTUAL BGE embeddings and nearest-neighbor recovery")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
