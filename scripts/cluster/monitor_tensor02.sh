#!/bin/bash

# Monitor tensor02 training progress
# Usage: ./scripts/cluster/monitor_tensor02.sh

set -euo pipefail

CLUSTER="tensor02"
SESSION="tensor02_test_fixed"

# Load credentials
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CRED_FILE="$REPO_ROOT/cluster_config/${CLUSTER}_credentials.json"

if [[ ! -f "$CRED_FILE" ]]; then
    echo "❌ Credentials file not found: $CRED_FILE"
    exit 1
fi

SERVER=$(jq -r '.server' "$CRED_FILE")
USERNAME=$(jq -r '.username' "$CRED_FILE")
PASSWORD=$(jq -r '.password' "$CRED_FILE")

echo "════════════════════════════════════════════════════════════"
echo "  TENSOR02 MONITORING - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
echo ""

# Function to run SSH command
run_ssh() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        ${USERNAME}@${SERVER} "$1" 2>&1 | grep -v "Warning: Permanently added"
}

# 1. Check if screen session exists
echo "🔍 Checking screen session..."
SCREEN_STATUS=$(run_ssh "screen -ls 2>&1")
if echo "$SCREEN_STATUS" | grep -q "$SESSION"; then
    echo "✅ Screen session '$SESSION' is running"
    SESSION_RUNNING=true
else
    echo "❌ Screen session '$SESSION' NOT found!"
    echo "$SCREEN_STATUS"
    SESSION_RUNNING=false
fi
echo ""

# 2. Check GPU usage
echo "🎮 GPU Status:"
run_ssh "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" | \
    awk -F',' '{printf "  GPU %s (%s): %s%% util, %s/%s MB, %s°C\n", $1, $2, $3, $4, $5, $6}'
echo ""

# 3. Check for training processes
echo "🔄 Training processes:"
PROCESSES=$(run_ssh "ps aux | grep '[p]ython.*train.py' | head -5")
if [[ -n "$PROCESSES" ]]; then
    echo "$PROCESSES" | awk '{printf "  PID %s: CPU %s%%, MEM %s%%, CMD: %s\n", $2, $3, $4, $11" "$12" "$13}'
    TRAINING_RUNNING=true
else
    echo "  ⚠️  No training processes found"
    TRAINING_RUNNING=false
fi
echo ""

# 4. Check latest log file
echo "📋 Latest logs:"
LATEST_LOG=$(run_ssh "ls -t ~/giblet-responses/logs/training_${SESSION}_*.log 2>/dev/null | head -1")
if [[ -n "$LATEST_LOG" ]]; then
    echo "  Log file: $LATEST_LOG"
    echo "  Last modified: $(run_ssh "stat -c '%y' $LATEST_LOG 2>/dev/null || stat -f '%Sm' $LATEST_LOG")"
    echo ""
    echo "  📄 Last 30 lines:"
    run_ssh "tail -30 $LATEST_LOG" | sed 's/^/    /'
    echo ""

    # Check for errors
    ERRORS=$(run_ssh "tail -100 $LATEST_LOG | grep -i 'error\|exception\|failed\|traceback' || true")
    if [[ -n "$ERRORS" ]]; then
        echo "  ⚠️  ERRORS DETECTED:"
        echo "$ERRORS" | sed 's/^/    /'
        echo ""
    fi

    # Extract epoch info if available
    EPOCH_INFO=$(run_ssh "tail -100 $LATEST_LOG | grep -E 'Epoch [0-9]+/[0-9]+' | tail -1 || true")
    if [[ -n "$EPOCH_INFO" ]]; then
        echo "  📊 Current progress: $EPOCH_INFO"
    fi

    # Extract loss info if available
    LOSS_INFO=$(run_ssh "tail -50 $LATEST_LOG | grep -E 'Loss:|loss:' | tail -5 || true")
    if [[ -n "$LOSS_INFO" ]]; then
        echo "  📉 Recent loss values:"
        echo "$LOSS_INFO" | sed 's/^/    /'
    fi
else
    echo "  ⚠️  No log files found matching pattern: training_${SESSION}_*.log"
fi
echo ""

# 5. Overall status assessment
echo "════════════════════════════════════════════════════════════"
echo "  STATUS SUMMARY"
echo "════════════════════════════════════════════════════════════"

if [[ "$SESSION_RUNNING" == true ]] && [[ "$TRAINING_RUNNING" == true ]]; then
    echo "✅ Training appears to be RUNNING NORMALLY"
elif [[ "$SESSION_RUNNING" == true ]] && [[ "$TRAINING_RUNNING" == false ]]; then
    echo "⚠️  Screen session exists but NO training process detected"
    echo "   This could mean:"
    echo "   - Training is still initializing"
    echo "   - Training crashed but screen is still open"
    echo "   - Training completed"
    echo ""
    echo "   💡 Action: Check the logs above for details"
elif [[ "$SESSION_RUNNING" == false ]]; then
    echo "❌ Screen session NOT found - training may have crashed or not started"
    echo ""
    echo "   💡 Action: Check logs and restart if needed:"
    echo "   ./scripts/cluster/remote_train.sh --cluster tensor02 \\"
    echo "       --config configs/training/tensor02_test_50epoch_config.yaml \\"
    echo "       --gpus 6 --name tensor02_test_fixed"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Monitoring complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
