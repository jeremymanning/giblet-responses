#!/bin/bash

# Remote Evaluation Script
# Runs model evaluation on remote GPU cluster and syncs results back

set -e

# Default values
CLUSTER="tensor02"
CHECKPOINT=""
CONFIG=""
OUTPUT_DIR="reconstruction_results"
NUM_SAMPLES=5
DEVICE="cuda"
SYNC_RESULTS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
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
        --no-sync)
            SYNC_RESULTS=false
            shift
            ;;
        --help)
            echo "Usage: $0 --cluster CLUSTER --checkpoint PATH --config PATH [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cluster NAME        Cluster to use (tensor01 or tensor02)"
            echo "  --checkpoint PATH     Remote path to checkpoint (relative to ~/giblet-responses/)"
            echo "  --config PATH         Local path to config YAML"
            echo "  --output-dir DIR      Output directory name (default: reconstruction_results)"
            echo "  --num-samples N       Number of samples to evaluate (default: 5)"
            echo "  --device DEVICE       Device to use: cpu or cuda (default: cuda)"
            echo "  --no-sync             Don't sync results back to local machine"
            echo "  --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --cluster tensor02 \\"
            echo "     --checkpoint tensor02_test_checkpoints/best_checkpoint.pt \\"
            echo "     --config configs/training/tensor02_test_50epoch_config.yaml \\"
            echo "     --num-samples 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ] || [ -z "$CONFIG" ]; then
    echo "Error: --checkpoint and --config are required"
    echo "Run with --help for usage information"
    exit 1
fi

# Cluster credentials
if [ "$CLUSTER" = "tensor01" ]; then
    HOST="f002d6b@tensor01.dartmouth.edu"
elif [ "$CLUSTER" = "tensor02" ]; then
    HOST="f002d6b@tensor02.dartmouth.edu"
else
    echo "Error: Unknown cluster '$CLUSTER'. Must be tensor01 or tensor02."
    exit 1
fi

PASSWORD="yaf1wue7gev_WQB.ueb"

echo "========================================="
echo "Remote Evaluation Setup"
echo "========================================="
echo "Cluster: $CLUSTER"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Num samples: $NUM_SAMPLES"
echo "Device: $DEVICE"
echo ""

# Upload config file to cluster
echo "Step 1: Uploading config file..."
sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$CONFIG" "$HOST:~/giblet-responses/$(basename $CONFIG)" || {
    echo "Error: Failed to upload config file"
    exit 1
}
echo "✓ Config uploaded"

# Create evaluation script on cluster
echo ""
echo "Step 2: Creating evaluation script on cluster..."

EVAL_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -e

CHECKPOINT="$1"
CONFIG="$2"
OUTPUT_DIR="$3"
NUM_SAMPLES="$4"
DEVICE="$5"

cd ~/giblet-responses

echo "========================================="
echo "Starting Model Evaluation"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Device: $DEVICE"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate giblet-py311

# Run evaluation
python scripts/evaluate_reconstructions.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --device "$DEVICE"

echo ""
echo "✓ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
EOF
)

# Upload and execute evaluation script
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$HOST" "cat > ~/giblet-responses/run_evaluation.sh" <<< "$EVAL_SCRIPT"

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$HOST" "chmod +x ~/giblet-responses/run_evaluation.sh"

echo "✓ Evaluation script created"

# Run evaluation
echo ""
echo "Step 3: Running evaluation on $CLUSTER..."
echo "This may take several minutes..."
echo ""

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$HOST" "cd ~/giblet-responses && ./run_evaluation.sh '$CHECKPOINT' '$(basename $CONFIG)' '$OUTPUT_DIR' '$NUM_SAMPLES' '$DEVICE'" || {
    echo ""
    echo "⚠️  Evaluation failed or was interrupted"
    echo "Check logs on cluster: ssh $HOST 'cat ~/giblet-responses/evaluation.log'"
    exit 1
}

# Sync results back if requested
if [ "$SYNC_RESULTS" = true ]; then
    echo ""
    echo "Step 4: Syncing results back to local machine..."

    LOCAL_OUTPUT_DIR="$OUTPUT_DIR"
    mkdir -p "$LOCAL_OUTPUT_DIR"

    sshpass -p "$PASSWORD" rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        "$HOST:~/giblet-responses/$OUTPUT_DIR/" "$LOCAL_OUTPUT_DIR/" || {
        echo "⚠️  Failed to sync results. Files are still available on cluster at:"
        echo "   $HOST:~/giblet-responses/$OUTPUT_DIR/"
        exit 1
    }

    echo ""
    echo "✓ Results synced to: $LOCAL_OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  ls -lh $LOCAL_OUTPUT_DIR/"
    echo "  open $LOCAL_OUTPUT_DIR/sample_1/"
else
    echo ""
    echo "✓ Evaluation complete (results not synced)"
    echo "Results available on cluster at:"
    echo "  $HOST:~/giblet-responses/$OUTPUT_DIR/"
fi

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
