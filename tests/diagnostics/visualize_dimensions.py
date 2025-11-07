"""
Visualize dimension flow through the autoencoder to identify bottleneck.
"""

import matplotlib.pyplot as plt

# Define layer dimensions
layers = [
    ("L1\nInput", 46272, "input"),
    ("L2A\nVideo", 1024, "encoder"),
    ("L2B\nAudio", 256, "encoder"),
    ("L2C\nText", 256, "encoder"),
    ("L3\nPool", 1536, "encoder"),
    ("L4\nConv", 1536, "encoder"),
    ("L5\nExpand", 4096, "encoder"),
    ("L6\nBottleneck", 8000, "bottleneck"),
    ("L7\nExpand", 2048, "decoder"),
    ("L8\nDeconv", 4096, "decoder"),
    ("L9\nUnpool", 8192, "decoder"),
    ("L10A\nVideo", 4096, "decoder"),
    ("L10B\nAudio", 1024, "decoder"),
    ("L10C\nText", 2048, "decoder"),
    ("L11\nOutput", 43200, "output"),
]

# Extract main pathway (excluding parallel branches)
main_pathway = [
    ("L1", 46272),
    ("L3", 1536),  # After pooling all branches
    ("L4", 1536),
    ("L5", 4096),
    ("L6", 8000),  # Supposed bottleneck
    ("L7", 2048),  # Actually smallest!
    ("L8", 4096),
    ("L9", 8192),
    ("L11", 43200),
]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: All layers
fig.suptitle(
    "Autoencoder Architecture: Dimension Flow Analysis", fontsize=16, fontweight="bold"
)

positions = []
dims = []
colors = []
labels = []

color_map = {
    "input": "#4CAF50",
    "encoder": "#2196F3",
    "bottleneck": "#FF5722",
    "decoder": "#9C27B0",
    "output": "#4CAF50",
}

for i, (label, dim, layer_type) in enumerate(layers):
    positions.append(i)
    dims.append(dim)
    colors.append(color_map[layer_type])
    labels.append(label)

bars = ax1.bar(
    positions, dims, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
)
ax1.set_xticks(positions)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Number of Dimensions", fontsize=12, fontweight="bold")
ax1.set_title("All Layers (Including Parallel Branches)", fontsize=14)
ax1.set_yscale("log")
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Highlight L6 and L7
for i, (label, dim, layer_type) in enumerate(layers):
    if "L6" in label:
        bars[i].set_edgecolor("red")
        bars[i].set_linewidth(3)
        ax1.text(
            i,
            dim,
            f"{dim:,}\n(L6)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="red",
        )
    elif "L7" in label:
        bars[i].set_edgecolor("darkgreen")
        bars[i].set_linewidth(3)
        ax1.text(
            i,
            dim,
            f"{dim:,}\n(TRUE MIN)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="darkgreen",
        )

# Plot 2: Main pathway only
main_positions = list(range(len(main_pathway)))
main_labels = [label for label, _ in main_pathway]
main_dims = [dim for _, dim in main_pathway]

# Color main pathway
main_colors = [
    (
        "#4CAF50"
        if i == 0 or i == len(main_pathway) - 1
        else (
            "#FF5722"
            if i == 5  # L6
            else (
                "#00C853"
                if i == 6  # L7 (true min)
                else "#2196F3"
                if i < 5
                else "#9C27B0"
            )
        )
    )
    for i in range(len(main_pathway))
]

bars2 = ax2.bar(
    main_positions,
    main_dims,
    color=main_colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)
ax2.set_xticks(main_positions)
ax2.set_xticklabels(main_labels, rotation=45, ha="right", fontsize=11)
ax2.set_ylabel("Number of Dimensions", fontsize=12, fontweight="bold")
ax2.set_title("Main Pathway (Sequential Flow)", fontsize=14)
ax2.set_yscale("log")
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# Draw lines connecting points
ax2.plot(
    main_positions, main_dims, "k--", alpha=0.4, linewidth=2, marker="o", markersize=8
)

# Annotate bottleneck issue
l6_idx = 5
l7_idx = 6
ax2.annotate(
    "",
    xy=(l7_idx, main_dims[l7_idx]),
    xytext=(l6_idx, main_dims[l6_idx]),
    arrowprops=dict(arrowstyle="<->", color="red", lw=2),
)
ax2.text(
    (l6_idx + l7_idx) / 2,
    (main_dims[l6_idx] + main_dims[l7_idx]) / 2,
    f"L7 < L6\n{main_dims[l7_idx]:,} < {main_dims[l6_idx]:,}",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
)

# Highlight L6
bars2[l6_idx].set_edgecolor("red")
bars2[l6_idx].set_linewidth(3)
ax2.text(
    l6_idx,
    main_dims[l6_idx],
    f"{main_dims[l6_idx]:,}\n(Spec: Smallest)",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="red",
)

# Highlight L7
bars2[l7_idx].set_edgecolor("darkgreen")
bars2[l7_idx].set_linewidth(3)
ax2.text(
    l7_idx,
    main_dims[l7_idx],
    f"{main_dims[l7_idx]:,}\n(Actually Smallest!)",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="darkgreen",
)

# Add legend
from matplotlib.patches import Patch  # noqa: E402

legend_elements = [
    Patch(facecolor="#4CAF50", edgecolor="black", label="Input/Output"),
    Patch(facecolor="#2196F3", edgecolor="black", label="Encoder"),
    Patch(facecolor="#FF5722", edgecolor="black", label="L6 (Spec Bottleneck)"),
    Patch(facecolor="#00C853", edgecolor="black", label="L7 (True Bottleneck)"),
    Patch(facecolor="#9C27B0", edgecolor="black", label="Decoder"),
]
ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(
    "/Users/jmanning/giblet-responses/notes/dimension_flow.png",
    dpi=300,
    bbox_inches="tight",
)
print("Saved dimension flow visualization to: notes/dimension_flow.png")

# Print summary
print("\n" + "=" * 80)
print("BOTTLENECK ANALYSIS SUMMARY")
print("=" * 80)
print("\nMain Pathway Dimensions:")
for label, dim in main_pathway:
    marker = (
        " ★ SPEC BOTTLENECK"
        if label == "L6"
        else " ★ TRUE BOTTLENECK"
        if label == "L7"
        else ""
    )
    print(f"  {label}: {dim:>7,}{marker}")

print("\nISSUE:")
print(f"  Layer 6 = {main_pathway[5][1]:,} (spec says smallest)")
print(f"  Layer 7 = {main_pathway[6][1]:,} (actually smallest!)")
print(f"  Ratio: L7/L6 = {main_pathway[6][1]/main_pathway[5][1]:.3f}")
print("\n  ⚠️  Layer 6 is LARGER than Layer 7, violating bottleneck principle!")
print("=" * 80)
