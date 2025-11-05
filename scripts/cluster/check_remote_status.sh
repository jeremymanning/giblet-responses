#!/bin/bash

# Detect project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

#===============================================================================
# check_remote_status.sh - Monitor remote cluster training status
#===============================================================================
# Usage: ./check_remote_status.sh --cluster CLUSTER_NAME
#
# Description:
#   Connects to a remote cluster and displays:
#   - Screen session status
#   - GPU utilization
#   - Recent training logs
#   - Disk usage
#
# Requirements:
#   - sshpass (for password-based SSH)
#   - jq (for JSON parsing)
#   - Cluster credentials in cluster_config/${CLUSTER}_credentials.json
#===============================================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${CYAN}${BOLD}$1${NC}"
}

print_subheader() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}$1${NC}"
}

# Print usage
usage() {
    cat << EOF
Usage: $0 --cluster CLUSTER_NAME

Monitor remote cluster training status.

Arguments:
    --cluster CLUSTER_NAME   Specify cluster (tensor01 or tensor02)
    --help                   Show this help message

Examples:
    $0 --cluster tensor01
    $0 --cluster tensor02

Requirements:
    - sshpass installed (brew install sshpass on macOS)
    - jq installed (brew install jq on macOS)
    - Cluster credentials in cluster_config/CLUSTER_credentials.json

EOF
    exit 1
}

# Parse command line arguments
CLUSTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$CLUSTER" ]; then
    print_error "Error: --cluster argument is required"
    usage
fi

# Validate cluster name
if [[ "$CLUSTER" != "tensor01" ]] && [[ "$CLUSTER" != "tensor02" ]]; then
    print_error "Error: Cluster must be 'tensor01' or 'tensor02'"
    exit 1
fi

# Check for required tools
command -v sshpass >/dev/null 2>&1 || {
    print_error "Error: sshpass is not installed"
    echo "  Install with: brew install sshpass"
    exit 1
}

command -v jq >/dev/null 2>&1 || {
    print_error "Error: jq is not installed"
    echo "  Install with: brew install jq"
    exit 1
}

command -v gnuplot >/dev/null 2>&1 || {
    print_warning "Warning: gnuplot is not installed (plotting disabled)"
    echo "  Install with: brew install gnuplot"
    GNUPLOT_AVAILABLE=false
}
if [ -z "$GNUPLOT_AVAILABLE" ]; then
    GNUPLOT_AVAILABLE=true
fi

# Load credentials
CRED_FILE="$PROJECT_ROOT/cluster_config/${CLUSTER}_credentials.json"

if [ ! -f "$CRED_FILE" ]; then
    print_error "Error: Credentials file not found: $CRED_FILE"
    exit 1
fi

# Parse credentials
SERVER=$(jq -r '.server' "$CRED_FILE")
USERNAME=$(jq -r '.username' "$CRED_FILE")
PASSWORD=$(jq -r '.password' "$CRED_FILE")
BASE_PATH=$(jq -r '.base_path' "$CRED_FILE")

if [ -z "$SERVER" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    print_error "Error: Invalid credentials file: $CRED_FILE"
    exit 1
fi

# Print header
echo ""
print_header "╔═══════════════════════════════════════╗"
print_header "║  Cluster Status: ${CLUSTER}$(printf ' %.0s' {1..14})║"
print_header "╚═══════════════════════════════════════╝"

# Test connection
print_subheader "Connection"
if sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${USERNAME}@${SERVER}" "exit" 2>/dev/null; then
    print_success "Connected to ${SERVER}"
else
    print_error "Failed to connect to ${SERVER}"
    exit 1
fi

# Check screen sessions
print_subheader "Screen Sessions"

SCREEN_OUTPUT=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" "screen -ls" 2>&1)

# Parse screen output
if echo "$SCREEN_OUTPUT" | grep -q "No Sockets found"; then
    print_warning "No screen sessions found"
    SESSION_COUNT=0
else
    # Extract session names and states
    SESSION_COUNT=$(echo "$SCREEN_OUTPUT" | grep -c "Detached\|Attached" 2>/dev/null || echo 0)

    if [ "$SESSION_COUNT" -gt 0 ]; then
        echo "$SCREEN_OUTPUT" | grep -E "^\s+[0-9]+\." | while read -r line; do
            # Extract session name and state
            SESSION_NAME=$(echo "$line" | awk '{print $2}' | cut -d'.' -f2)
            SESSION_STATE=$(echo "$line" | grep -o "(Detached\|Attached)" | tr -d '()')

            if [ -n "$SESSION_NAME" ]; then
                print_success "$SESSION_NAME ($SESSION_STATE)"
            fi
        done
        echo "Total: $SESSION_COUNT active session$([ "$SESSION_COUNT" -ne 1 ] && echo "s" || echo "")"
    else
        print_warning "No screen sessions found"
    fi
fi

# Check GPU utilization
print_subheader "GPU Utilization"

GPU_OUTPUT=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>&1)

if [ $? -eq 0 ]; then
    ACTIVE_GPU_COUNT=0

    while IFS=, read -r INDEX NAME UTIL MEM_USED MEM_TOTAL; do
        # Clean up values
        INDEX=$(echo "$INDEX" | xargs)
        NAME=$(echo "$NAME" | xargs)
        UTIL=$(echo "$UTIL" | xargs | sed 's/ %//')
        MEM_USED=$(echo "$MEM_USED" | xargs | sed 's/ MiB//')
        MEM_TOTAL=$(echo "$MEM_TOTAL" | xargs | sed 's/ MiB//')

        # Convert to GB for display
        MEM_USED_GB=$(echo "scale=1; $MEM_USED / 1024" | bc)
        MEM_TOTAL_GB=$(echo "scale=1; $MEM_TOTAL / 1024" | bc)

        # Format utilization percentage
        UTIL_FORMATTED=$(printf "%3s" "$UTIL")

        # Check if GPU is active (>5% util or >1GB memory)
        IS_ACTIVE=0
        if [ "$UTIL" -gt 5 ] 2>/dev/null || [ "$MEM_USED" -gt 1024 ] 2>/dev/null; then
            IS_ACTIVE=1
            ACTIVE_GPU_COUNT=$((ACTIVE_GPU_COUNT + 1))
        fi

        # Print GPU status with highlighting for active GPUs
        if [ "$IS_ACTIVE" -eq 1 ]; then
            echo -e "GPU ${INDEX}: ${NAME} | Util: ${BOLD}${UTIL_FORMATTED}%${NC} | Mem: ${BOLD}${MEM_USED_GB}GB / ${MEM_TOTAL_GB}GB${NC} ${YELLOW}⚡${NC}"
        else
            echo "GPU ${INDEX}: ${NAME} | Util: ${UTIL_FORMATTED}% | Mem: ${MEM_USED_GB}GB / ${MEM_TOTAL_GB}GB"
        fi
    done <<< "$GPU_OUTPUT"

    if [ "$ACTIVE_GPU_COUNT" -gt 0 ]; then
        echo -e "\n${GREEN}${ACTIVE_GPU_COUNT}${NC} GPU$([ "$ACTIVE_GPU_COUNT" -ne 1 ] && echo "s" || echo "") actively training"
    fi
else
    print_error "Failed to query GPU status"
fi

# Check training logs
print_subheader "Recent Training Logs"

LOGS_OUTPUT=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
    "cd ${BASE_PATH} && if [ -d logs ]; then ls -lt logs/*.log 2>/dev/null | head -n 3; else echo 'NO_LOGS'; fi" 2>&1)

if [ "$LOGS_OUTPUT" = "NO_LOGS" ]; then
    print_warning "No logs directory or log files found"
else
    LOG_COUNT=$(echo "$LOGS_OUTPUT" | grep -c "\.log" || echo 0)

    if [ "$LOG_COUNT" -gt 0 ]; then
        INDEX=1
        echo "$LOGS_OUTPUT" | while read -r PERMS LINKS OWNER GROUP SIZE MONTH DAY TIME FILENAME; do
            if [[ "$FILENAME" == *.log ]]; then
                # Extract just the filename
                BASENAME=$(basename "$FILENAME")

                # Format size
                if [[ "$SIZE" =~ ^[0-9]+$ ]]; then
                    SIZE_MB=$(echo "scale=1; $SIZE / 1048576" | bc)
                    SIZE_DISPLAY="${SIZE_MB}MB"
                else
                    SIZE_DISPLAY="$SIZE"
                fi

                # Calculate time ago
                CURRENT_TIME=$(date +%s)
                LOG_TIME=$(date -j -f "%b %d %H:%M" "$MONTH $DAY $TIME" +%s 2>/dev/null || echo "$CURRENT_TIME")
                TIME_DIFF=$((CURRENT_TIME - LOG_TIME))

                if [ "$TIME_DIFF" -lt 3600 ]; then
                    TIME_AGO="$((TIME_DIFF / 60)) min ago"
                elif [ "$TIME_DIFF" -lt 86400 ]; then
                    TIME_AGO="$((TIME_DIFF / 3600)) hr ago"
                else
                    TIME_AGO="$((TIME_DIFF / 86400)) days ago"
                fi

                echo "${INDEX}. ${BASENAME} (${SIZE_DISPLAY}) - ${TIME_AGO}"
                INDEX=$((INDEX + 1))
            fi
        done

        # Check latest log for errors
        LATEST_LOG=$(echo "$LOGS_OUTPUT" | grep "\.log" | head -n 1 | awk '{print $NF}')

        if [ -n "$LATEST_LOG" ]; then
            echo -e "\n${BLUE}Latest log tail (last 20 lines, errors only):${NC}"

            LATEST_ERRORS=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
                "cd ${BASE_PATH} && tail -n 20 ${LATEST_LOG} 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -i 'error\|exception\|failed\|traceback' || echo 'NO_ERRORS'" 2>&1)

            if [ "$LATEST_ERRORS" = "NO_ERRORS" ]; then
                print_success "No errors detected in recent log entries"
            else
                echo "$LATEST_ERRORS" | while read -r line; do
                    print_warning "$line"
                done
            fi
        fi
    else
        print_warning "No .log files found in logs/ directory"
    fi
fi

# Check disk usage
print_subheader "Disk Usage"

# Get all directory sizes in a single SSH session to avoid connection issues
DISK_SIZES=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
    "cd ${BASE_PATH} && du -sh data/ checkpoints/ logs/ test_checkpoints/ test_logs/ . 2>/dev/null" 2>&1 | grep -v "Warning:")

if [ -n "$DISK_SIZES" ]; then
    # Parse the output (format: "SIZE<tab>path/")
    DATA_SIZE=$(echo "$DISK_SIZES" | awk '/\tdata\// {print $1}')
    CHECKPOINTS_SIZE=$(echo "$DISK_SIZES" | awk '/\tcheckpoints\// {print $1}')
    LOGS_SIZE=$(echo "$DISK_SIZES" | awk '/\tlogs\// {print $1}')
    TEST_CHECKPOINTS_SIZE=$(echo "$DISK_SIZES" | awk '/\ttest_checkpoints\// {print $1}')
    TEST_LOGS_SIZE=$(echo "$DISK_SIZES" | awk '/\ttest_logs\// {print $1}')
    TOTAL_SIZE=$(echo "$DISK_SIZES" | awk '/\t\.$/ {print $1}')

    # Use fallback values if any are empty
    [ -z "$DATA_SIZE" ] && DATA_SIZE="0B"
    [ -z "$CHECKPOINTS_SIZE" ] && CHECKPOINTS_SIZE="0B"
    [ -z "$LOGS_SIZE" ] && LOGS_SIZE="0B"
    [ -z "$TEST_CHECKPOINTS_SIZE" ] && TEST_CHECKPOINTS_SIZE="0B"
    [ -z "$TEST_LOGS_SIZE" ] && TEST_LOGS_SIZE="0B"
    [ -z "$TOTAL_SIZE" ] && TOTAL_SIZE="0B"

    echo "data/: $DATA_SIZE"
    echo "checkpoints/: $CHECKPOINTS_SIZE"
    echo "logs/: $LOGS_SIZE"
    echo "test_checkpoints/: $TEST_CHECKPOINTS_SIZE"
    echo "test_logs/: $TEST_LOGS_SIZE"
    echo "Total: $TOTAL_SIZE"
else
    print_warning "Unable to determine disk usage"
fi

# Training Progress Analysis
if [ -n "$LATEST_LOG" ]; then
    print_subheader "Training Progress"

    # Fetch full log for analysis
    LOG_CONTENT=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
        "cd ${BASE_PATH} && cat ${LATEST_LOG} 2>/dev/null" 2>&1)

    if [ $? -eq 0 ] && [ -n "$LOG_CONTENT" ]; then
        # Parse log to extract epoch info
        # Format: Epoch N/TOTAL | Train Loss: X.XXXX | Val Loss: X.XXXX | ...

        EPOCH_LINES=$(echo "$LOG_CONTENT" | grep -E "^Epoch [0-9]+/[0-9]+" | tail -50)

        if [ -n "$EPOCH_LINES" ]; then
            # Extract current and total epochs
            LAST_EPOCH_LINE=$(echo "$EPOCH_LINES" | tail -1)
            CURRENT_EPOCH=$(echo "$LAST_EPOCH_LINE" | grep -oE "^Epoch [0-9]+" | grep -oE "[0-9]+")
            TOTAL_EPOCHS=$(echo "$LAST_EPOCH_LINE" | grep -oE "Epoch [0-9]+/[0-9]+" | grep -oE "/[0-9]+" | tr -d '/')

            if [ -n "$CURRENT_EPOCH" ] && [ -n "$TOTAL_EPOCHS" ]; then
                EPOCHS_REMAINING=$((TOTAL_EPOCHS - CURRENT_EPOCH))
                PROGRESS_PCT=$(echo "scale=1; $CURRENT_EPOCH * 100 / $TOTAL_EPOCHS" | bc)

                echo -e "${BOLD}Current Epoch:${NC} $CURRENT_EPOCH / $TOTAL_EPOCHS (${PROGRESS_PCT}%)"
                echo -e "${BOLD}Epochs Remaining:${NC} $EPOCHS_REMAINING"

                # Estimate time per epoch and completion time
                # Parse timestamps from log entries
                # Most recent 20 epochs for better accuracy
                RECENT_EPOCHS=$(echo "$EPOCH_LINES" | tail -20)

                # Create temp files for analysis
                TEMP_DIR=$(mktemp -d)
                EPOCH_TIMES_FILE="$TEMP_DIR/epoch_times.txt"

                # Extract epoch numbers and their timestamps
                echo "$RECENT_EPOCHS" > "$TEMP_DIR/recent_epochs.txt"

                # Get log modification times for each epoch line using grep with line numbers
                # This is approximate - we'll estimate based on log file modification
                FIRST_EPOCH_IN_LOG=$(echo "$EPOCH_LINES" | head -1 | grep -oE "^Epoch [0-9]+" | grep -oE "[0-9]+")
                EPOCHS_IN_LOG=$(echo "$EPOCH_LINES" | wc -l | xargs)

                if [ "$EPOCHS_IN_LOG" -ge 2 ]; then
                    # Get log file's age
                    LOG_START_TIME=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
                        "cd ${BASE_PATH} && stat -f %B ${LATEST_LOG} 2>/dev/null || stat -c %Y ${LATEST_LOG} 2>/dev/null" 2>&1)

                    CURRENT_TIME=$(date +%s)

                    if [ -n "$LOG_START_TIME" ] && [ "$LOG_START_TIME" -gt 0 ]; then
                        # Total time elapsed for logged epochs
                        TIME_ELAPSED=$((CURRENT_TIME - LOG_START_TIME))

                        # Estimate time per epoch
                        TIME_PER_EPOCH=$(echo "scale=2; $TIME_ELAPSED / $EPOCHS_IN_LOG" | bc)

                        # Convert to minutes
                        TIME_PER_EPOCH_MIN=$(echo "scale=1; $TIME_PER_EPOCH / 60" | bc)

                        # Estimate remaining time
                        TIME_REMAINING_SEC=$(echo "$TIME_PER_EPOCH * $EPOCHS_REMAINING" | bc)
                        TIME_REMAINING_MIN=$(echo "scale=0; $TIME_REMAINING_SEC / 60" | bc)
                        TIME_REMAINING_HR=$(echo "scale=1; $TIME_REMAINING_MIN / 60" | bc)

                        # Calculate 95% confidence interval (assuming ~20% variability)
                        # This is a rough estimate based on typical training variance
                        STDERR=$(echo "scale=2; $TIME_PER_EPOCH * 0.10" | bc)  # 10% std error
                        CI_MARGIN=$(echo "scale=2; $STDERR * 1.96" | bc)  # 95% CI

                        TIME_MIN=$(echo "scale=2; ($TIME_PER_EPOCH - $CI_MARGIN) * $EPOCHS_REMAINING / 3600" | bc)
                        TIME_MAX=$(echo "scale=2; ($TIME_PER_EPOCH + $CI_MARGIN) * $EPOCHS_REMAINING / 3600" | bc)

                        # Ensure min time is not negative
                        if [ $(echo "$TIME_MIN < 0" | bc) -eq 1 ]; then
                            TIME_MIN=0
                        fi

                        echo -e "\n${BOLD}Time per Epoch:${NC} ${TIME_PER_EPOCH_MIN} minutes"

                        if [ "$TIME_REMAINING_HR" -lt 2 ]; then
                            echo -e "${BOLD}Estimated Remaining:${NC} ${TIME_REMAINING_MIN} minutes"
                        else
                            echo -e "${BOLD}Estimated Remaining:${NC} ${TIME_REMAINING_HR} hours"
                        fi

                        echo -e "${BOLD}95% Confidence Range:${NC} ${TIME_MIN} - ${TIME_MAX} hours"

                        # Calculate estimated completion time
                        COMPLETION_TIMESTAMP=$((CURRENT_TIME + TIME_REMAINING_SEC))
                        COMPLETION_TIME=$(date -r "$COMPLETION_TIMESTAMP" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "@$COMPLETION_TIMESTAMP" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)

                        if [ -n "$COMPLETION_TIME" ]; then
                            echo -e "${BOLD}Estimated Completion:${NC} ${COMPLETION_TIME}"
                        fi
                    else
                        print_warning "Could not determine log start time"
                    fi
                else
                    print_warning "Not enough epochs logged for time estimation (need at least 2)"
                fi

                # Clean up temp files
                rm -rf "$TEMP_DIR"

                # Create loss plot if gnuplot is available
                if [ "$GNUPLOT_AVAILABLE" = true ]; then
                    print_subheader "Training Loss Plot"

                    # Create temp files for plotting
                    PLOT_DIR=$(mktemp -d)
                    TRAIN_LOSS_FILE="$PLOT_DIR/train_loss.dat"
                    VAL_LOSS_FILE="$PLOT_DIR/val_loss.dat"
                    PLOT_SCRIPT="$PLOT_DIR/plot.gp"

                    # Extract loss data from all epoch lines
                    # Format: epoch train_loss val_loss
                    echo "$EPOCH_LINES" | while read -r line; do
                        EPOCH=$(echo "$line" | grep -oE "^Epoch [0-9]+" | grep -oE "[0-9]+")
                        TRAIN_LOSS=$(echo "$line" | grep -oE "Train Loss: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
                        VAL_LOSS=$(echo "$line" | grep -oE "Val Loss: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")

                        if [ -n "$EPOCH" ] && [ -n "$TRAIN_LOSS" ]; then
                            echo "$EPOCH $TRAIN_LOSS" >> "$TRAIN_LOSS_FILE"
                        fi

                        if [ -n "$EPOCH" ] && [ -n "$VAL_LOSS" ]; then
                            echo "$EPOCH $VAL_LOSS" >> "$VAL_LOSS_FILE"
                        fi
                    done

                    # Check if we have data to plot
                    TRAIN_COUNT=$(wc -l < "$TRAIN_LOSS_FILE" 2>/dev/null || echo 0)
                    VAL_COUNT=$(wc -l < "$VAL_LOSS_FILE" 2>/dev/null || echo 0)

                    if [ "$TRAIN_COUNT" -gt 0 ]; then
                        # Create gnuplot script
                        cat > "$PLOT_SCRIPT" << 'PLOTEOF'
set terminal dumb size 120,30
set title "Training Progress"
set xlabel "Epoch"
set ylabel "Loss"
set key outside right top
set grid
PLOTEOF

                        # Add plot command based on available data
                        if [ "$VAL_COUNT" -gt 0 ]; then
                            echo "plot '$TRAIN_LOSS_FILE' using 1:2 with lines title 'Train Loss', '$VAL_LOSS_FILE' using 1:2 with lines title 'Val Loss'" >> "$PLOT_SCRIPT"
                        else
                            echo "plot '$TRAIN_LOSS_FILE' using 1:2 with lines title 'Train Loss'" >> "$PLOT_SCRIPT"
                        fi

                        # Generate plot
                        gnuplot "$PLOT_SCRIPT" 2>/dev/null

                        if [ $? -ne 0 ]; then
                            print_warning "Failed to generate plot"
                        fi
                    else
                        print_warning "No loss data available for plotting"
                    fi

                    # Clean up temp files
                    rm -rf "$PLOT_DIR"
                fi
            else
                print_warning "Could not parse epoch information from log"
            fi
        else
            print_warning "No epoch information found in log file"
        fi
    else
        print_warning "Could not read log file for analysis"
    fi
fi

# Summary
print_subheader "Summary"

# Count active training sessions (screen sessions)
if [ "$SESSION_COUNT" -gt 0 ]; then
    print_success "$SESSION_COUNT training session$([ "$SESSION_COUNT" -ne 1 ] && echo "s" || echo "") active"
else
    print_warning "No training sessions active"
fi

# GPU status
if [ "$ACTIVE_GPU_COUNT" -gt 0 ]; then
    print_success "$ACTIVE_GPU_COUNT GPU$([ "$ACTIVE_GPU_COUNT" -ne 1 ] && echo "s" || echo "") actively training"
else
    print_warning "No GPUs actively training"
fi

# Log errors
if [ -n "$LATEST_LOG" ]; then
    LATEST_ERRORS=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${SERVER}" \
        "cd ${BASE_PATH} && tail -n 20 ${LATEST_LOG} 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -i 'error\|exception\|failed\|traceback' || echo 'NO_ERRORS'" 2>&1)

    if [ "$LATEST_ERRORS" = "NO_ERRORS" ]; then
        print_success "No errors detected in recent logs"
    else
        ERROR_COUNT=$(echo "$LATEST_ERRORS" | wc -l | xargs)
        print_warning "$ERROR_COUNT error$([ "$ERROR_COUNT" -ne 1 ] && echo "s" || echo "") detected in recent logs"
    fi
fi

echo ""
