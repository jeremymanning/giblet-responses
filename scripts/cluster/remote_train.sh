#!/bin/bash

# Detect project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# remote_train.sh - Remote training script for giblet-responses project
# Manages training jobs on tensor01/tensor02 clusters via screen sessions
#
# This script handles:
#   - SSH authentication via sshpass using cluster credentials
#   - Code synchronization via GitHub (git pull)
#   - Remote environment setup verification
#   - Training job launch in persistent screen sessions (NO SLURM)
#   - Multi-GPU training via torchrun (handled by run_giblet.sh)
#
# Usage: ./remote_train.sh --cluster CLUSTER_NAME [OPTIONS]
# Run with --help for full documentation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GPUS=8
SESSION_NAME="giblet_train"
DRY_RUN=false
RESUME=false
KILL=false
CLUSTER=""
CONFIG=""

# Print usage information
usage() {
    cat << EOF
Usage: $0 --cluster CLUSTER_NAME [OPTIONS]

Required arguments:
  --cluster CLUSTER_NAME    Cluster to use (tensor01 or tensor02)

Optional arguments:
  --config CONFIG_PATH      Path to training config YAML file
  --gpus NUM_GPUS          Number of GPUs to use (1-8, default: 8)
  --resume                 Resume training from checkpoint
  --kill                   Kill existing training session before starting
  --name SESSION_NAME      Name for screen session (default: giblet_train)
  --dry-run                Show commands without executing
  --help                   Show this help message

Examples:
  # Start training with default config on tensor01
  $0 --cluster tensor01

  # Start training with custom config on tensor02 using 4 GPUs
  $0 --cluster tensor02 --config configs/my_config.yaml --gpus 4

  # Resume training from checkpoint
  $0 --cluster tensor01 --resume

  # Kill existing session and start new training
  $0 --cluster tensor01 --kill

  # Dry run to see what would be executed
  $0 --cluster tensor01 --dry-run

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --kill)
            KILL=true
            shift
            ;;
        --name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown argument: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CLUSTER" ]]; then
    echo -e "${RED}Error: --cluster is required${NC}"
    usage
fi

if [[ "$CLUSTER" != "tensor01" && "$CLUSTER" != "tensor02" ]]; then
    echo -e "${RED}Error: Cluster must be tensor01 or tensor02${NC}"
    exit 1
fi

# Validate GPU count
if [[ $GPUS -lt 1 || $GPUS -gt 8 ]]; then
    echo -e "${RED}Error: Number of GPUs must be between 1 and 8${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}=== Remote Training Configuration ===${NC}"
echo -e "Cluster:       ${GREEN}$CLUSTER${NC}"
echo -e "GPUs:          ${GREEN}$GPUS${NC}"
echo -e "Session Name:  ${GREEN}$SESSION_NAME${NC}"
echo -e "Config:        ${GREEN}${CONFIG:-default}${NC}"
echo -e "Resume:        ${GREEN}$RESUME${NC}"
echo -e "Kill Existing: ${GREEN}$KILL${NC}"
echo -e "Dry Run:       ${GREEN}$DRY_RUN${NC}"
echo ""

# Load credentials
CRED_FILE="$PROJECT_ROOT/cluster_config/${CLUSTER}_credentials.json"
if [[ ! -f "$CRED_FILE" ]]; then
    echo -e "${RED}Error: Credential file not found: $CRED_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}=== Loading Credentials ===${NC}"
SERVER=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])")
USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])")
PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])")
BASE_PATH=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['base_path'])")

echo -e "Server:   ${GREEN}$SERVER${NC}"
echo -e "Username: ${GREEN}$USERNAME${NC}"
echo -e "Path:     ${GREEN}$BASE_PATH${NC}"
echo ""

# Check for sshpass
if ! command -v sshpass &> /dev/null; then
    echo -e "${YELLOW}Warning: sshpass not found${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Installing sshpass via Homebrew...${NC}"
        if ! $DRY_RUN; then
            brew install hudochenkov/sshpass/sshpass
        else
            echo -e "${BLUE}[DRY RUN]${NC} Would run: brew install hudochenkov/sshpass/sshpass"
        fi
    else
        echo -e "${RED}Error: Please install sshpass manually${NC}"
        exit 1
    fi
fi

# Helper function to run SSH commands
run_ssh() {
    local cmd="$1"
    if $DRY_RUN; then
        echo -e "${BLUE}[DRY RUN]${NC} Would run SSH: $cmd"
    else
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$SERVER" "$cmd"
    fi
}

# Helper function to check SSH connection
test_ssh_connection() {
    echo -e "${BLUE}=== Testing SSH Connection ===${NC}"
    if $DRY_RUN; then
        echo -e "${BLUE}[DRY RUN]${NC} Would test SSH connection"
        return 0
    fi

    if sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$USERNAME@$SERVER" "echo 'Connection successful'" &> /dev/null; then
        echo -e "${GREEN}SSH connection successful${NC}"
        return 0
    else
        echo -e "${RED}SSH connection failed${NC}"
        return 1
    fi
}

# Test connection
if ! test_ssh_connection; then
    exit 1
fi
echo ""

# Check for existing screen sessions
echo -e "${BLUE}=== Checking for Existing Sessions ===${NC}"
if $DRY_RUN; then
    echo -e "${BLUE}[DRY RUN]${NC} Would check for existing sessions named: $SESSION_NAME"
    if $KILL; then
        echo -e "${BLUE}[DRY RUN]${NC} Would kill any existing sessions"
    fi
else
    EXISTING_SESSIONS=$(run_ssh "screen -ls | grep -c '$SESSION_NAME' || true")
    if [[ "$EXISTING_SESSIONS" != "0" ]] && [[ "$EXISTING_SESSIONS" != "" ]]; then
        echo -e "${YELLOW}Found existing session: $SESSION_NAME${NC}"

        if $KILL; then
            echo -e "${YELLOW}Killing existing session...${NC}"
            run_ssh "screen -X -S $SESSION_NAME quit || true"
            sleep 2
        else
            echo -e "${YELLOW}Use --kill to terminate existing session${NC}"
            echo -e "${YELLOW}Or use a different session name with --name${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}No existing session found${NC}"
    fi
fi
echo ""

# Backup checkpoints if resuming
if $RESUME; then
    echo -e "${BLUE}=== Backing Up Checkpoints ===${NC}"
    BACKUP_DIR="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
    if ! $DRY_RUN; then
        run_ssh "cd $BASE_PATH && if [ -d checkpoints ]; then cp -r checkpoints $BACKUP_DIR; echo 'Backed up to $BACKUP_DIR'; else echo 'No checkpoints to backup'; fi"
    else
        echo -e "${BLUE}[DRY RUN]${NC} Would backup checkpoints to $BACKUP_DIR"
    fi
    echo ""
fi

# Sync code from GitHub instead of rsync
echo -e "${BLUE}=== Synchronizing Code via GitHub ===${NC}"

SETUP_CMD="cd $BASE_PATH && source ~/.bashrc && source ~/giblet-responses/scripts/cluster/setup_cluster.sh && setup_cluster_environment"

if $DRY_RUN; then
    echo -e "${BLUE}[DRY RUN]${NC} Would run GitHub sync on cluster"
    echo -e "${BLUE}[DRY RUN]${NC} Command: $SETUP_CMD"
else
    echo -e "Running: ${GREEN}GitHub sync${NC}"
    run_ssh "$SETUP_CMD"
fi
echo ""

# Prepare training command
TRAIN_CMD="cd $BASE_PATH && source ~/.bashrc && conda activate giblet-py311 && "
TRAIN_CMD+="./run_giblet.sh --task train"
if [[ -n "$CONFIG" ]]; then
    TRAIN_CMD+=" --config $CONFIG"
fi
TRAIN_CMD+=" --gpus $GPUS"
if $RESUME; then
    TRAIN_CMD+=" --resume"
fi

# Create logs directory
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${SESSION_NAME}_${TIMESTAMP}.log"

# Launch training job in screen session
echo -e "${BLUE}=== Launching Training Job ===${NC}"
SCREEN_CMD="screen -dmS $SESSION_NAME bash -c '"
SCREEN_CMD+="source ~/.bashrc; "
SCREEN_CMD+="conda activate giblet-py311; "
SCREEN_CMD+="cd $BASE_PATH; "
SCREEN_CMD+="mkdir -p $LOG_DIR; "
SCREEN_CMD+="$TRAIN_CMD 2>&1 | tee $LOG_FILE"
SCREEN_CMD+="'"

if $DRY_RUN; then
    echo -e "${BLUE}[DRY RUN]${NC} Would create screen session: $SESSION_NAME"
    echo -e "${BLUE}[DRY RUN]${NC} Command: $TRAIN_CMD"
    echo -e "${BLUE}[DRY RUN]${NC} Log file: $LOG_FILE"
else
    echo -e "Creating screen session: ${GREEN}$SESSION_NAME${NC}"
    run_ssh "$SCREEN_CMD"
    sleep 2

    # Verify screen session was created
    SESSION_CHECK=$(run_ssh "screen -ls | grep -c '$SESSION_NAME' || true")
    if [[ "$SESSION_CHECK" == "1" ]] || [[ "$SESSION_CHECK" =~ ^[1-9] ]]; then
        echo -e "${GREEN}Training job launched successfully!${NC}"
    else
        echo -e "${RED}Warning: Could not verify screen session was created${NC}"
    fi
fi
echo ""

# Print post-launch instructions
echo -e "${GREEN}=== Training Job Launched ===${NC}"
echo ""
echo -e "${BLUE}To attach to the training session:${NC}"
echo -e "  ${GREEN}ssh $USERNAME@$SERVER${NC}"
echo -e "  ${GREEN}screen -r $SESSION_NAME${NC}"
echo ""
echo -e "${BLUE}To detach from screen session:${NC}"
echo -e "  ${GREEN}Press Ctrl+A, then D${NC}"
echo ""
echo -e "${BLUE}To check GPU utilization:${NC}"
echo -e "  ${GREEN}ssh $USERNAME@$SERVER${NC}"
echo -e "  ${GREEN}nvidia-smi${NC}"
echo ""
echo -e "${BLUE}To view training logs:${NC}"
echo -e "  ${GREEN}ssh $USERNAME@$SERVER${NC}"
echo -e "  ${GREEN}cd $BASE_PATH && tail -f $LOG_FILE${NC}"
echo ""
echo -e "${BLUE}To sync results back to local machine:${NC}"
echo -e "  ${GREEN}sshpass -p [PASSWORD] rsync -avz --progress $USERNAME@$SERVER:$BASE_PATH/checkpoints/ ./checkpoints/${NC}"
echo -e "  ${GREEN}sshpass -p [PASSWORD] rsync -avz --progress $USERNAME@$SERVER:$BASE_PATH/logs/ ./logs/${NC}"
echo ""
echo -e "${BLUE}To kill the training session:${NC}"
echo -e "  ${GREEN}$0 --cluster $CLUSTER --name $SESSION_NAME --kill${NC}"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}=== DRY RUN COMPLETE ===${NC}"
    echo -e "${YELLOW}No commands were actually executed${NC}"
fi
