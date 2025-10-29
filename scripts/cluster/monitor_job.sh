#!/bin/bash
# Monitor job status on SLURM cluster
# Usage: ./monitor_job.sh <tensor01|tensor02> <job_id> [--tail] [--error]

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
JOB_ID=""
TAIL_MODE=false
ERROR_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tail)
            TAIL_MODE=true
            shift
            ;;
        --error)
            ERROR_MODE=true
            shift
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER=$1
            elif [ -z "$JOB_ID" ]; then
                JOB_ID=$1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$CLUSTER" ] || [ -z "$JOB_ID" ]; then
    print_error "Usage: $0 <tensor01|tensor02> <job_id> [--tail] [--error]"
    echo ""
    echo "Options:"
    echo "  --tail     Follow log output (like tail -f)"
    echo "  --error    Show error log instead of output log"
    echo ""
    echo "Examples:"
    echo "  $0 tensor01 12345"
    echo "  $0 tensor02 67890 --tail"
    echo "  $0 tensor01 12345 --error --tail"
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

print_success "Credentials loaded"
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    print_error "sshpass not found. Please install it first."
    exit 1
fi

print_info "Monitoring job $JOB_ID on $SERVER"
echo ""

# Get job status
print_info "Fetching job status..."
echo ""

JOB_STATUS=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "squeue -j $JOB_ID -o '%i %T %P %N %C %m %l %M' 2>/dev/null || echo 'NOT_FOUND'" 2>/dev/null)

if [ "$JOB_STATUS" = "NOT_FOUND" ]; then
    print_warning "Job not found in queue. It may have completed."

    # Try to get job info from sacct
    COMPLETED_STATUS=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "$USERNAME@$SERVER" "sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,NodeList,Elapsed --noheader 2>/dev/null || echo 'NOT_FOUND'" 2>/dev/null)

    if [ "$COMPLETED_STATUS" != "NOT_FOUND" ]; then
        echo -e "${CYAN}Job History:${NC}"
        echo "$COMPLETED_STATUS"
    fi
else
    echo -e "${CYAN}Current Job Status:${NC}"
    echo "JOBID STATE PARTITION NODES CPUS MEMORY TIME ELAPSED"
    echo "$JOB_STATUS"
fi

echo ""

# Get job details from sacct
print_info "Fetching detailed job information..."
DETAILED_INFO=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "sacct -j $JOB_ID --format=JobID,JobName,State,User,Account,Start,End,Elapsed,CPUTime,MaxRSS,ExitCode --noheader 2>/dev/null || echo ''" 2>/dev/null)

if [ -n "$DETAILED_INFO" ]; then
    echo ""
    echo -e "${CYAN}Detailed Information:${NC}"
    echo "$DETAILED_INFO"
fi

echo ""

# Find and display logs
print_info "Searching for log files..."

FIND_LOGS=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "find $BASE_PATH/slurm_logs -name \"${JOB_ID}*\" 2>/dev/null | head -10 || echo ''" 2>/dev/null)

if [ -n "$FIND_LOGS" ]; then
    echo ""
    echo -e "${CYAN}Log files found:${NC}"
    echo "$FIND_LOGS"

    # Determine which log to show
    if [ "$ERROR_MODE" = true ]; then
        LOG_TYPE="error"
        LOG_FILE=$(echo "$FIND_LOGS" | grep "\.err$" | head -1)
    else
        LOG_TYPE="output"
        LOG_FILE=$(echo "$FIND_LOGS" | grep "\.out$" | head -1)
    fi

    if [ -n "$LOG_FILE" ]; then
        echo ""
        echo -e "${CYAN}Displaying ${LOG_TYPE} log:${NC}"
        print_info "Log file: $LOG_FILE"
        echo ""

        if [ "$TAIL_MODE" = true ]; then
            print_info "Following log (Ctrl+C to exit)..."
            echo ""
            sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
                "$USERNAME@$SERVER" "tail -f $LOG_FILE" 2>/dev/null
        else
            # Get last 50 lines
            print_info "Last 50 lines of log:"
            echo ""
            sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
                "$USERNAME@$SERVER" "tail -50 $LOG_FILE" 2>/dev/null

            echo ""
            echo -e "${YELLOW}(To follow log in real-time, use: $0 $CLUSTER $JOB_ID --tail)${NC}"
        fi
    else
        print_warning "Could not locate log file"
    fi
else
    print_warning "No log files found yet. Job may still be initializing."
fi

echo ""
print_success "Monitoring complete"
echo ""
