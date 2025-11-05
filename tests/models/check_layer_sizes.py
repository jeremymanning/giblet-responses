"""Check all layer sizes in the architecture to verify bottleneck is smallest."""

from giblet.models.autoencoder import MultimodalAutoencoder

model = MultimodalAutoencoder(
    video_height=90,
    video_width=160,
    audio_mels=2048,
    text_dim=1024,
    n_voxels=85810,
    bottleneck_dim=8000,
)

# Collect all layer sizes
layer_sizes = []

# Encoder
encoder = model.encoder
layer_sizes.append(("Layer 1 (video input)", 90 * 160 * 3))
layer_sizes.append(("Layer 1 (audio input)", 2048))
layer_sizes.append(("Layer 1 (text input)", 1024))
layer_sizes.append(("Layer 1 (total input)", 90 * 160 * 3 + 2048 + 1024))

layer_sizes.append(("Layer 2A (video flat)", encoder.video_encoder.flat_features))
layer_sizes.append(("Layer 2A (video output)", encoder.video_encoder.output_features))

layer_sizes.append(("Layer 2B (audio flat)", encoder.audio_encoder.flat_features))
layer_sizes.append(("Layer 2B (audio output)", encoder.audio_encoder.output_features))

layer_sizes.append(("Layer 2C (text intermediate)", 512))
layer_sizes.append(("Layer 2C (text output)", encoder.text_encoder.output_features))

layer_sizes.append(("Layer 3 (pooled)", encoder.pooled_dim))

layer_sizes.append(("Layer 4 (feature conv output)", encoder.pooled_dim))

layer_sizes.append(("Layer 5 (intermediate)", 4096))
layer_sizes.append(("Layer 6 (BOTTLENECK)", encoder.bottleneck_dim))

layer_sizes.append(("Layer 5 expansion to voxels", encoder.n_voxels))

# Decoder
decoder = model.decoder
layer_sizes.append(("Layer 7 (expand)", decoder.hidden_dim))
layer_sizes.append(("Layer 8 (deconv)", decoder.hidden_dim * 2))
layer_sizes.append(("Layer 9 (unpool)", decoder.hidden_dim * 4))
layer_sizes.append(("Layer 10A (video)", decoder.hidden_dim * 2))
layer_sizes.append(("Layer 10B (audio)", decoder.hidden_dim // 2))
layer_sizes.append(("Layer 10C (text)", decoder.hidden_dim))
layer_sizes.append(("Layer 11 (video output)", decoder.video_dim))
layer_sizes.append(("Layer 11 (audio output)", decoder.audio_dim))
layer_sizes.append(("Layer 11 (text output)", decoder.text_dim))

# Sort by size
sorted_layers = sorted(layer_sizes, key=lambda x: x[1])

print("All layers sorted by size (smallest to largest):")
print("=" * 80)
for name, size in sorted_layers:
    marker = (
        " ★ SMALLEST ★"
        if size == encoder.bottleneck_dim and name.startswith("Layer 6")
        else ""
    )
    print(f"{size:>10,}  {name}{marker}")

print("\n" + "=" * 80)
print(f"Layer 6 bottleneck: {encoder.bottleneck_dim:,}")
print(f"Is smallest layer? {encoder.bottleneck_dim == sorted_layers[0][1]}")

# Find all layers smaller than bottleneck
print("\nLayers SMALLER than Layer 6 bottleneck (8,000):")
for name, size in sorted_layers:
    if size < encoder.bottleneck_dim:
        print(f"  {size:>10,}  {name}")
