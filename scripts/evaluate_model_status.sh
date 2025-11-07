#!/bin/bash
#
# Model Evaluation and Status Tool
# =================================
#
# This script provides comprehensive evaluation of trained models including:
# - Checkpoint status and metadata
# - Model weight health checks (NaN/Inf detection)
# - Reconstruction quality visualization
# - Performance metrics summary
#
# Usage:
#   ./scripts/evaluate_model_status.sh [options]
#
# Options:
#   --checkpoint PATH       Path to checkpoint file (required)
#   --config PATH          Path to training config YAML (required)
#   --output-dir PATH      Output directory for results (default: evaluation_results)
#   --num-samples N        Number of test samples to evaluate (default: 5)
#   --device DEVICE        Device to use: cpu or cuda (default: cpu)
#   --skip-weights         Skip weight health check
#   --skip-reconstructions Skip reconstruction visualization
#   --help                 Show this help message
#
# Examples:
#   # Full evaluation with default settings
#   ./scripts/evaluate_model_status.sh \
#       --checkpoint checkpoints/best_model.pt \
#       --config configs/training/config.yaml
#
#   # Quick checkpoint status check only
#   ./scripts/evaluate_model_status.sh \
#       --checkpoint checkpoints/best_model.pt \
#       --config configs/training/config.yaml \
#       --skip-weights --skip-reconstructions
#
#   # Detailed reconstruction analysis
#   ./scripts/evaluate_model_status.sh \
#       --checkpoint checkpoints/best_model.pt \
#       --config configs/training/config.yaml \
#       --num-samples 20 \
#       --output-dir detailed_evaluation
#

set -e  # Exit on error

# Default values
CHECKPOINT=""
CONFIG=""
OUTPUT_DIR="evaluation_results"
NUM_SAMPLES=5
DEVICE="cpu"
SKIP_WEIGHTS=false
SKIP_RECONSTRUCTIONS=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "$1"
}

show_help() {
    sed -n '/^#/!q;s/^# //;s/^#//;3,$p' "$0"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-weights)
            SKIP_WEIGHTS=true
            shift
            ;;
        --skip-reconstructions)
            SKIP_RECONSTRUCTIONS=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_help
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    print_error "Missing required argument: --checkpoint"
    echo ""
    show_help
fi

if [ -z "$CONFIG" ]; then
    print_error "Missing required argument: --config"
    echo ""
    show_help
fi

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    print_error "Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Validate config exists
if [ ! -f "$CONFIG" ]; then
    print_error "Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start evaluation
print_header "Model Evaluation Status Check"
echo ""
print_info "Checkpoint: $CHECKPOINT"
print_info "Config: $CONFIG"
print_info "Output Directory: $OUTPUT_DIR"
print_info "Device: $DEVICE"
echo ""

# Step 1: Checkpoint metadata
print_header "1. Checkpoint Metadata"
echo ""

CHECKPOINT_SIZE=$(du -h "$CHECKPOINT" | cut -f1)
CHECKPOINT_DATE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$CHECKPOINT" 2>/dev/null || stat -c "%y" "$CHECKPOINT" 2>/dev/null | cut -d' ' -f1,2)

print_info "File size: $CHECKPOINT_SIZE"
print_info "Last modified: $CHECKPOINT_DATE"
echo ""

# Load checkpoint info using Python
python3 << EOF
import torch
import sys

try:
    checkpoint = torch.load("$CHECKPOINT", map_location='cpu')

    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")

    if 'train_loss' in checkpoint:
        print(f"Train Loss: {checkpoint['train_loss']:.4f}")

    if 'val_loss' in checkpoint:
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")

    if 'best_val_loss' in checkpoint:
        print(f"Best Validation Loss: {checkpoint['best_val_loss']:.4f}")

    # Count parameters
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        print("Warning: No model state dict found")
        sys.exit(1)

    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"Total Parameters: {total_params:,}")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Checkpoint loaded successfully"
else
    print_error "Failed to load checkpoint"
    exit 1
fi
echo ""

# Step 2: Weight health check
if [ "$SKIP_WEIGHTS" = false ]; then
    print_header "2. Weight Health Check"
    echo ""

    if [ -f "scripts/examine_weights.py" ]; then
        python scripts/examine_weights.py "$CHECKPOINT" > "$OUTPUT_DIR/weight_health_report.txt" 2>&1

        # Check for critical issues
        if grep -q "Layers with NaN: 0" "$OUTPUT_DIR/weight_health_report.txt" && \
           grep -q "Layers with Inf: 0" "$OUTPUT_DIR/weight_health_report.txt"; then
            print_success "No NaN or Inf values detected in model weights"
        else
            print_warning "Weight health issues detected - see $OUTPUT_DIR/weight_health_report.txt"
        fi

        echo ""
        print_info "Full weight health report saved to: $OUTPUT_DIR/weight_health_report.txt"
    else
        print_warning "Weight examination script not found (scripts/examine_weights.py)"
    fi
    echo ""
else
    print_info "Skipping weight health check (--skip-weights)"
    echo ""
fi

# Step 3: Reconstruction quality visualization
if [ "$SKIP_RECONSTRUCTIONS" = false ]; then
    print_header "3. Reconstruction Quality Evaluation"
    echo ""

    print_info "Generating reconstructions for $NUM_SAMPLES test samples..."
    print_info "This may take several minutes..."
    echo ""

    RECON_DIR="$OUTPUT_DIR/reconstructions"

    if [ -f "scripts/evaluate_reconstructions.py" ]; then
        python scripts/evaluate_reconstructions.py \
            --checkpoint "$CHECKPOINT" \
            --config "$CONFIG" \
            --output-dir "$RECON_DIR" \
            --num-samples "$NUM_SAMPLES" \
            --device "$DEVICE" 2>&1 | tee "$OUTPUT_DIR/reconstruction_log.txt"

        if [ $? -eq 0 ]; then
            print_success "Reconstruction evaluation complete"
            print_info "Results saved to: $RECON_DIR"

            # Count generated files
            NUM_SAMPLES_GENERATED=$(find "$RECON_DIR" -name "sample_*" -type d | wc -l | tr -d ' ')
            print_info "Samples evaluated: $NUM_SAMPLES_GENERATED"

            # List sample directories
            echo ""
            print_info "Generated visualizations:"
            ls -d "$RECON_DIR"/sample_* 2>/dev/null | while read sample_dir; do
                sample_num=$(basename "$sample_dir")
                echo "  - $sample_num/"
                ls "$sample_dir"/*.png 2>/dev/null | while read png; do
                    echo "      $(basename "$png")"
                done
            done
        else
            print_error "Reconstruction evaluation failed"
            print_info "Check log: $OUTPUT_DIR/reconstruction_log.txt"
        fi
    else
        print_warning "Reconstruction evaluation script not found (scripts/evaluate_reconstructions.py)"
    fi
    echo ""
else
    print_info "Skipping reconstruction evaluation (--skip-reconstructions)"
    echo ""
fi

# Step 4: Summary
print_header "4. Evaluation Summary"
echo ""

print_info "Evaluation complete! Results saved to: $OUTPUT_DIR"
echo ""

print_info "Generated files:"
if [ -f "$OUTPUT_DIR/weight_health_report.txt" ]; then
    echo "  - weight_health_report.txt"
fi
if [ -f "$OUTPUT_DIR/reconstruction_log.txt" ]; then
    echo "  - reconstruction_log.txt"
fi
if [ -d "$OUTPUT_DIR/reconstructions" ]; then
    echo "  - reconstructions/ (visualization images)"
fi

echo ""
print_success "Model evaluation completed successfully!"
