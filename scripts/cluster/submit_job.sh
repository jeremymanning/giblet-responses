#!/bin/bash
# Submit training job to SLURM cluster (tensor01 or tensor02)
# Usage: ./submit_job.sh <tensor01|tensor02> <job_name> <script_path> [python_args...]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

# Validate arguments
if [ $# -lt 3 ]; then
    print_error "Usage: $0 <tensor01|tensor02> <job_name> <script_path> [python_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 tensor01 training1 train.py --epochs 100"
    echo "  $0 tensor02 decoder demo_decoder.py --model bert-base-uncased"
    exit 1
fi

CLUSTER=$1
JOB_NAME=$2
SCRIPT_PATH=$3
shift 3
PYTHON_ARGS="$@"

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
GPUS=$(read_json "$CREDS_FILE" "gpus")
GPU_TYPE=$(read_json "$CREDS_FILE" "gpu_type")

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ] || [ -z "$SERVER" ]; then
    print_error "Failed to read credentials from JSON file"
    exit 1
fi

# Set defaults if not specified
GPUS=${GPUS:-8}
GPU_TYPE=${GPU_TYPE:-A6000}

print_success "Credentials loaded for $SERVER"
echo ""

# Verify script exists locally
if [ ! -f "$PROJECT_ROOT/$SCRIPT_PATH" ]; then
    print_error "Script not found: $PROJECT_ROOT/$SCRIPT_PATH"
    exit 1
fi

print_info "Target cluster: $CLUSTER ($SERVER)"
print_info "Job name: $JOB_NAME"
print_info "Script: $SCRIPT_PATH"
print_info "GPUs requested: $GPUS x $GPU_TYPE"
if [ -n "$PYTHON_ARGS" ]; then
    print_info "Python arguments: $PYTHON_ARGS"
fi
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    print_error "sshpass not found. Please install it first."
    exit 1
fi

# Create SLURM submission script
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SLURM_SCRIPT="/tmp/submit_${JOB_NAME}_${TIMESTAMP}.sh"
REMOTE_SLURM_SCRIPT="${BASE_PATH}/slurm_jobs/submit_${JOB_NAME}_${TIMESTAMP}.sh"

# Create local SLURM script
cat > "$SLURM_SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=$GPUS
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=${BASE_PATH}/slurm_logs/%j_${JOB_NAME}.out
#SBATCH --error=${BASE_PATH}/slurm_logs/%j_${JOB_NAME}.err

# Setup environment
set -e
source ~/.bashrc
conda activate giblet-env

# Navigate to project directory
cd $BASE_PATH

# Run the training script
echo "Starting job: $JOB_NAME"
echo "Timestamp: \$(date)"
echo "Node: \$(hostname)"
echo "GPU Count: \$(nvidia-smi --list-gpus | wc -l)"
echo ""

python $SCRIPT_PATH $PYTHON_ARGS

echo ""
echo "Job completed: $JOB_NAME"
echo "Timestamp: \$(date)"
SLURM_EOF

chmod +x "$SLURM_SCRIPT"

print_info "SLURM script created: $SLURM_SCRIPT"
echo ""

# Copy SLURM script to cluster and create directories
print_info "Preparing cluster for job submission..."

SETUP_DIRS_SCRIPT=$(cat <<'EOF'
mkdir -p $BASE_PATH/slurm_jobs
mkdir -p $BASE_PATH/slurm_logs
echo "Directories ready"
EOF
)

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "bash -c 'BASE_PATH=\"$BASE_PATH\" bash -s' <<'SETUP_EOF'
$SETUP_DIRS_SCRIPT
SETUP_EOF
" 2>/dev/null || {
    print_warning "Failed to create directories on cluster"
}

# Copy SLURM script to cluster
print_info "Copying SLURM script to cluster..."
export SSHPASS="$PASSWORD"
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$SLURM_SCRIPT" "$USERNAME@$SERVER:$REMOTE_SLURM_SCRIPT" 2>/dev/null || {
    print_error "Failed to copy SLURM script to cluster"
    exit 1
}
print_success "SLURM script copied"
echo ""

# Submit job
print_info "Submitting job to SLURM..."
echo ""

SUBMIT_OUTPUT=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "chmod +x $REMOTE_SLURM_SCRIPT && sbatch $REMOTE_SLURM_SCRIPT" 2>/dev/null)

if [ -z "$SUBMIT_OUTPUT" ]; then
    print_error "Failed to submit job"
    exit 1
fi

# Extract job ID from output
JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP '(?<=job )\d+' || echo "")

if [ -z "$JOB_ID" ]; then
    print_warning "Could not parse job ID from response"
    print_info "Raw output: $SUBMIT_OUTPUT"
else
    print_success "Job submitted successfully!"
    echo ""
    echo "=========================================="
    echo "Job Information:"
    echo "  Cluster: $CLUSTER"
    echo "  Job ID: $JOB_ID"
    echo "  Job Name: $JOB_NAME"
    echo "  Server: $SERVER"
    echo "=========================================="
    echo ""
    echo "Monitor the job:"
    echo "  $SCRIPT_DIR/monitor_job.sh $CLUSTER $JOB_ID"
    echo ""
    echo "View logs:"
    echo "  sshpass -p \"$PASSWORD\" ssh $USERNAME@$SERVER tail -f $BASE_PATH/slurm_logs/${JOB_ID}_${JOB_NAME}.out"
    echo ""

    # Save job info
    JOB_INFO_FILE="$PROJECT_ROOT/.job_info_${CLUSTER}_${JOB_ID}.txt"
    cat > "$JOB_INFO_FILE" <<JOB_INFO_EOF
Cluster: $CLUSTER
Server: $SERVER
Job ID: $JOB_ID
Job Name: $JOB_NAME
Script: $SCRIPT_PATH
Python Args: $PYTHON_ARGS
Submit Time: $(date)
SLURM Script: $REMOTE_SLURM_SCRIPT
Output Log: $BASE_PATH/slurm_logs/${JOB_ID}_${JOB_NAME}.out
Error Log: $BASE_PATH/slurm_logs/${JOB_ID}_${JOB_NAME}.err
JOB_INFO_EOF

    print_success "Job info saved to: $JOB_INFO_FILE"
fi

echo ""

# Cleanup
rm -f "$SLURM_SCRIPT"
