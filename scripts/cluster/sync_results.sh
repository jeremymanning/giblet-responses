#!/bin/bash
# Sync results and checkpoints from cluster to local machine
# Usage: ./sync_results.sh <tensor01|tensor02> [--dry-run]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Function to print messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to read JSON value
read_json() {
    local json_file=$1
    local key=$2
    python3 -c "import json; data=json.load(open('$json_file')); print(data.get('$key', ''))"
}

# Parse arguments
CLUSTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER=$1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$CLUSTER" ]; then
    print_error "Usage: $0 <tensor01|tensor02> [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 tensor01"
    echo "  $0 tensor02 --dry-run"
    exit 1
fi

# Validate cluster
if [[ "$CLUSTER" != "tensor01" && "$CLUSTER" != "tensor02" ]]; then
    print_error "Invalid cluster name. Must be 'tensor01' or 'tensor02'"
    exit 1
fi

# Read credentials from JSON
CREDS_FILE="$PROJECT_ROOT/cluster_config/${CLUSTER}_credentials.json"
if [ ! -f "$CREDS_FILE" ]; then
    print_error "Credentials file not found: $CREDS_FILE"
    exit 1
fi

print_info "Reading credentials from $CREDS_FILE"
USERNAME=$(read_json "$CREDS_FILE" "username")
PASSWORD=$(read_json "$CREDS_FILE" "password")
SERVER=$(read_json "$CREDS_FILE" "server")
BASE_PATH=$(read_json "$CREDS_FILE" "base_path")

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ] || [ -z "$SERVER" ]; then
    print_error "Failed to read credentials from JSON file"
    exit 1
fi

print_success "Credentials loaded for $SERVER"
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    print_error "sshpass not found. Please install it first."
    exit 1
fi

print_info "Syncing results from cluster: $CLUSTER ($SERVER)"
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No files will be copied"
fi
echo ""

# Define local and remote directories
REMOTE_CHECKPOINTS="$BASE_PATH/checkpoints"
REMOTE_RESULTS="$BASE_PATH/results"
REMOTE_LOGS="$BASE_PATH/slurm_logs"
REMOTE_OUTPUT="$BASE_PATH/output"

LOCAL_CHECKPOINTS="$PROJECT_ROOT/checkpoints_${CLUSTER}"
LOCAL_RESULTS="$PROJECT_ROOT/results_${CLUSTER}"
LOCAL_LOGS="$PROJECT_ROOT/logs_${CLUSTER}"
LOCAL_OUTPUT="$PROJECT_ROOT/output_${CLUSTER}"

# Create local directories
print_info "Creating local directories..."
mkdir -p "$LOCAL_CHECKPOINTS"
mkdir -p "$LOCAL_RESULTS"
mkdir -p "$LOCAL_LOGS"
mkdir -p "$LOCAL_OUTPUT"
print_success "Local directories ready"
echo ""

# Setup rsync with sshpass
export SSHPASS="$PASSWORD"

# Function to sync directory with progress
sync_directory() {
    local remote_dir=$1
    local local_dir=$2
    local dir_name=$3

    print_info "Syncing $dir_name..."
    print_info "  Remote: $remote_dir"
    print_info "  Local: $local_dir"

    # First check if remote directory exists
    REMOTE_EXISTS=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "$USERNAME@$SERVER" "test -d $remote_dir && echo 'yes' || echo 'no'" 2>/dev/null)

    if [ "$REMOTE_EXISTS" = "no" ]; then
        print_warning "Remote directory does not exist: $remote_dir (skipping)"
        return
    fi

    # Build rsync command
    RSYNC_CMD="rsync -e 'sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' -avz --progress --stats $USERNAME@$SERVER:$remote_dir/ $local_dir/"

    if [ "$DRY_RUN" = true ]; then
        RSYNC_CMD="$RSYNC_CMD --dry-run"
    fi

    # Execute rsync
    if eval "$RSYNC_CMD" 2>/dev/null; then
        print_success "$dir_name synced successfully"
    else
        print_warning "Some issues encountered while syncing $dir_name (may be normal if directory is empty)"
    fi

    echo ""
}

# Sync directories
sync_directory "$REMOTE_CHECKPOINTS" "$LOCAL_CHECKPOINTS" "Checkpoints"
sync_directory "$REMOTE_RESULTS" "$LOCAL_RESULTS" "Results"
sync_directory "$REMOTE_LOGS" "$LOCAL_LOGS" "SLURM Logs"
sync_directory "$REMOTE_OUTPUT" "$LOCAL_OUTPUT" "Output"

# Show summary
echo ""
echo -e "${CYAN}=========================================="
echo "Sync Summary"
echo "==========================================${NC}"
echo ""
print_info "Local checkpoints: $LOCAL_CHECKPOINTS"
if [ -d "$LOCAL_CHECKPOINTS" ] && [ "$(ls -A $LOCAL_CHECKPOINTS 2>/dev/null)" ]; then
    echo "  Files: $(find $LOCAL_CHECKPOINTS -type f | wc -l)"
    echo "  Size: $(du -sh $LOCAL_CHECKPOINTS | cut -f1)"
else
    echo "  (empty)"
fi
echo ""

print_info "Local results: $LOCAL_RESULTS"
if [ -d "$LOCAL_RESULTS" ] && [ "$(ls -A $LOCAL_RESULTS 2>/dev/null)" ]; then
    echo "  Files: $(find $LOCAL_RESULTS -type f | wc -l)"
    echo "  Size: $(du -sh $LOCAL_RESULTS | cut -f1)"
else
    echo "  (empty)"
fi
echo ""

print_info "Local logs: $LOCAL_LOGS"
if [ -d "$LOCAL_LOGS" ] && [ "$(ls -A $LOCAL_LOGS 2>/dev/null)" ]; then
    echo "  Files: $(find $LOCAL_LOGS -type f | wc -l)"
    echo "  Size: $(du -sh $LOCAL_LOGS | cut -f1)"
else
    echo "  (empty)"
fi
echo ""

print_info "Local output: $LOCAL_OUTPUT"
if [ -d "$LOCAL_OUTPUT" ] && [ "$(ls -A $LOCAL_OUTPUT 2>/dev/null)" ]; then
    echo "  Files: $(find $LOCAL_OUTPUT -type f | wc -l)"
    echo "  Size: $(du -sh $LOCAL_OUTPUT | cut -f1)"
else
    echo "  (empty)"
fi
echo ""

# Check for additional files to sync
print_info "Checking for additional files to sync..."
CUSTOM_FILES=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "find $BASE_PATH -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' -o -name '*.txt' -o -name '*.log' \) 2>/dev/null | head -20 || echo ''" 2>/dev/null)

if [ -n "$CUSTOM_FILES" ]; then
    echo -e "${CYAN}Additional files found on cluster:${NC}"
    echo "$CUSTOM_FILES"
    echo ""
    print_warning "To sync these files, copy them to checkpoints/, results/, or output/ directories"
fi

echo ""
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN COMPLETE - No files were actually copied"
else
    print_success "Sync complete!"
fi
echo ""
print_info "Next steps:"
echo "  - Review synced files in $PROJECT_ROOT"
echo "  - For real-time syncing: $SCRIPT_DIR/sync_results.sh $CLUSTER"
echo "  - For dry run: $SCRIPT_DIR/sync_results.sh $CLUSTER --dry-run"
echo ""
