#!/bin/bash
#
# sync_checkpoints.sh - Download checkpoints and logs from remote clusters
#
# Purpose: Synchronize trained models and logs from tensor01/tensor02 to local machine
#
# Usage:
#   ./sync_checkpoints.sh --cluster CLUSTER_NAME --checkpoint-dir DIR [OPTIONS]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CLUSTER=""
CHECKPOINT_DIR=""
LOGS_ONLY=false
INCLUDE_LOGS=false
DRY_RUN=false

# Print usage
usage() {
    cat << EOF
Usage: $0 --cluster CLUSTER_NAME [OPTIONS]

Required arguments:
  --cluster CLUSTER_NAME    Cluster to sync from (tensor01 or tensor02)

Optional arguments:
  --checkpoint-dir DIR      Checkpoint directory to download
  --logs-only               Download logs only (no checkpoints)
  --include-logs            Also download logs (when downloading checkpoints)
  --dry-run                 Show what would be downloaded without downloading
  --help                    Show this help message

Examples:
  # Download specific checkpoint directory
  $0 --cluster tensor01 --checkpoint-dir production_500epoch_checkpoints

  # Download checkpoint directory and logs
  $0 --cluster tensor01 --checkpoint-dir test_8gpu_checkpoints --include-logs

  # Download logs only
  $0 --cluster tensor01 --logs-only

  # Dry run to see what would be downloaded
  $0 --cluster tensor01 --checkpoint-dir production_500epoch_checkpoints --dry-run

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
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --logs-only)
            LOGS_ONLY=true
            shift
            ;;
        --include-logs)
            INCLUDE_LOGS=true
            shift
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

if [[ "$LOGS_ONLY" == false && -z "$CHECKPOINT_DIR" ]]; then
    echo -e "${RED}Error: Either --checkpoint-dir or --logs-only is required${NC}"
    usage
fi

# Print configuration
echo -e "${BLUE}=== Checkpoint Synchronization ===${NC}"
echo -e "Cluster:         ${GREEN}$CLUSTER${NC}"
if [[ "$LOGS_ONLY" == true ]]; then
    echo -e "Mode:            ${GREEN}Logs only${NC}"
else
    echo -e "Checkpoint dir:  ${GREEN}$CHECKPOINT_DIR${NC}"
    echo -e "Include logs:    ${GREEN}$INCLUDE_LOGS${NC}"
fi
echo -e "Dry run:         ${GREEN}$DRY_RUN${NC}"
echo ""

# Load credentials
CRED_FILE="cluster_config/${CLUSTER}_credentials.json"
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
            echo -e "${BLUE}[DRY RUN]${NC} Would install sshpass"
        fi
    else
        echo -e "${RED}Error: Please install sshpass manually${NC}"
        exit 1
    fi
fi

# Download checkpoints
if [[ "$LOGS_ONLY" == false ]]; then
    echo -e "${BLUE}=== Downloading Checkpoints ===${NC}"

    # Create local checkpoint directory
    LOCAL_CHECKPOINT_DIR="./$CHECKPOINT_DIR"
    mkdir -p "$LOCAL_CHECKPOINT_DIR"

    REMOTE_CHECKPOINT_PATH="$USERNAME@$SERVER:$BASE_PATH/$CHECKPOINT_DIR/"

    if $DRY_RUN; then
        echo -e "${BLUE}[DRY RUN]${NC} Would download from:"
        echo -e "  Remote: $REMOTE_CHECKPOINT_PATH"
        echo -e "  Local:  $LOCAL_CHECKPOINT_DIR"
    else
        echo -e "Downloading from: ${GREEN}$REMOTE_CHECKPOINT_PATH${NC}"
        echo -e "To: ${GREEN}$LOCAL_CHECKPOINT_DIR${NC}"
        echo ""

        sshpass -p "$PASSWORD" rsync -avz --progress "$REMOTE_CHECKPOINT_PATH" "$LOCAL_CHECKPOINT_DIR/"

        if [[ $? -eq 0 ]]; then
            echo ""
            echo -e "${GREEN}Checkpoints downloaded successfully${NC}"

            # Show what was downloaded
            echo ""
            echo -e "${BLUE}Downloaded files:${NC}"
            ls -lh "$LOCAL_CHECKPOINT_DIR"
        else
            echo -e "${RED}Failed to download checkpoints${NC}"
            exit 1
        fi
    fi

    echo ""
fi

# Download logs
if [[ "$LOGS_ONLY" == true || "$INCLUDE_LOGS" == true ]]; then
    echo -e "${BLUE}=== Downloading Logs ===${NC}"

    # Create local logs directory
    LOCAL_LOGS_DIR="./logs"
    mkdir -p "$LOCAL_LOGS_DIR"

    REMOTE_LOGS_PATH="$USERNAME@$SERVER:$BASE_PATH/logs/"

    if $DRY_RUN; then
        echo -e "${BLUE}[DRY RUN]${NC} Would download from:"
        echo -e "  Remote: $REMOTE_LOGS_PATH"
        echo -e "  Local:  $LOCAL_LOGS_DIR"
    else
        echo -e "Downloading from: ${GREEN}$REMOTE_LOGS_PATH${NC}"
        echo -e "To: ${GREEN}$LOCAL_LOGS_DIR${NC}"
        echo ""

        sshpass -p "$PASSWORD" rsync -avz --progress "$REMOTE_LOGS_PATH" "$LOCAL_LOGS_DIR/"

        if [[ $? -eq 0 ]]; then
            echo ""
            echo -e "${GREEN}Logs downloaded successfully${NC}"

            # Show recent logs
            echo ""
            echo -e "${BLUE}Recent log files:${NC}"
            ls -lht "$LOCAL_LOGS_DIR" | head -10
        else
            echo -e "${RED}Failed to download logs${NC}"
            exit 1
        fi
    fi

    echo ""
fi

# Final summary
echo -e "${GREEN}=== Synchronization Complete ===${NC}"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}This was a dry run - no files were actually downloaded${NC}"
    echo -e "${YELLOW}Remove --dry-run to download files${NC}"
else
    echo ""
    if [[ "$LOGS_ONLY" == false ]]; then
        echo -e "Checkpoints: ${GREEN}$LOCAL_CHECKPOINT_DIR${NC}"
    fi
    if [[ "$LOGS_ONLY" == true || "$INCLUDE_LOGS" == true ]]; then
        echo -e "Logs:        ${GREEN}$LOCAL_LOGS_DIR${NC}"
    fi
fi

echo ""
