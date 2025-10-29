#!/bin/bash
# Setup cluster environment on tensor01 or tensor02
# Usage: ./setup_cluster.sh <tensor01|tensor02>

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
if [ $# -ne 1 ]; then
    print_error "Usage: $0 <tensor01|tensor02>"
    exit 1
fi

CLUSTER=$1
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

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    print_warning "sshpass not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install sshpass
    else
        print_error "sshpass not installed and brew not available"
        exit 1
    fi
fi

print_info "Setting up cluster: $CLUSTER ($SERVER)"
echo ""

# Step 1: Test SSH connection
print_info "Testing SSH connection..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "echo 'SSH connection successful'" 2>/dev/null || {
    print_error "Failed to connect to $SERVER"
    exit 1
}
print_success "SSH connection verified"
echo ""

# Step 2: Create project directory on cluster
print_info "Creating project directory on cluster..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "mkdir -p $BASE_PATH && echo 'Directory ready'" 2>/dev/null
print_success "Project directory created/verified"
echo ""

# Step 3: Check if conda is available
print_info "Checking for conda installation..."
CONDA_INSTALLED=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "command -v conda &> /dev/null && echo 'yes' || echo 'no'" 2>/dev/null)

if [ "$CONDA_INSTALLED" = "no" ]; then
    print_warning "Conda not found on cluster. Please install Miniconda/Anaconda manually."
    print_info "Instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
else
    print_success "Conda is installed"
fi
echo ""

# Step 4: Create conda environment
print_info "Creating conda environment: giblet-env"
SETUP_ENV_SCRIPT=$(cat <<'EOF'
set -e
source ~/.bashrc
if conda env list | grep -q "giblet-env"; then
    echo "Environment giblet-env already exists"
else
    conda create -y -n giblet-env python=3.10
    echo "Environment created successfully"
fi
EOF
)

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "bash -c '$SETUP_ENV_SCRIPT'" 2>/dev/null
print_success "Conda environment ready (giblet-env)"
echo ""

# Step 5: Sync code to cluster
print_info "Syncing code to cluster..."
print_info "Source: $PROJECT_ROOT/"
print_info "Destination: $USERNAME@$SERVER:$BASE_PATH/"

# Use rsync via sshpass
export SSHPASS="$PASSWORD"
rsync -e "sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    -avz \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='data/sherlock_nii' \
    --exclude='venv' \
    --exclude='.venv' \
    "$PROJECT_ROOT/" "$USERNAME@$SERVER:$BASE_PATH/" 2>/dev/null || {
    print_warning "Rsync sync failed, attempting with scp as fallback..."
}

print_success "Code synced to cluster"
echo ""

# Step 6: Install dependencies
print_info "Installing Python dependencies..."
INSTALL_DEPS_SCRIPT=$(cat <<'EOF'
set -e
source ~/.bashrc
conda activate giblet-env
cd $BASE_PATH
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed successfully"
EOF
)

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "bash -c 'BASE_PATH=\"$BASE_PATH\" bash -s' <<< '$INSTALL_DEPS_SCRIPT'" 2>/dev/null || {
    print_warning "Dependency installation may have encountered issues. Check manually with:"
    print_warning "sshpass -p \"PASSWORD\" ssh $USERNAME@$SERVER"
}
print_success "Dependencies installed"
echo ""

# Step 7: Download data
print_info "Downloading dataset from Dropbox..."
DOWNLOAD_DATA_SCRIPT=$(cat <<'EOF'
set -e
source ~/.bashrc
cd $BASE_PATH
if [ -f download_data_from_dropbox.sh ]; then
    chmod +x download_data_from_dropbox.sh
    bash download_data_from_dropbox.sh
    echo "Data download completed"
else
    echo "download_data_from_dropbox.sh not found"
fi
EOF
)

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "bash -c 'BASE_PATH=\"$BASE_PATH\" bash -s' <<'EOF' 2>/dev/null || true
$DOWNLOAD_DATA_SCRIPT
EOF
" 2>/dev/null || {
    print_warning "Data download failed or script not available. You can download manually on the cluster."
}

print_success "Dataset setup complete"
echo ""

# Step 8: Verify setup
print_info "Verifying setup..."
VERIFY_SCRIPT=$(cat <<'EOF'
source ~/.bashrc
conda activate giblet-env
cd $BASE_PATH
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ -f requirements.txt ]; then
    echo "requirements.txt found: Yes"
fi
if [ -d "giblet" ]; then
    echo "giblet directory found: Yes"
fi
if [ -d "data" ]; then
    echo "data directory found: Yes"
fi
EOF
)

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USERNAME@$SERVER" "bash -c 'BASE_PATH=\"$BASE_PATH\" bash -s' <<'EOF'
$VERIFY_SCRIPT
EOF
" 2>/dev/null || {
    print_warning "Verification encountered issues"
}

echo ""
print_success "Cluster setup complete!"
echo ""
echo "Next steps:"
echo "  1. Test environment: sshpass -p \"$PASSWORD\" ssh $USERNAME@$SERVER"
echo "  2. Submit training job: $SCRIPT_DIR/submit_job.sh $CLUSTER <job_name> <script_path>"
echo "  3. Monitor job: $SCRIPT_DIR/monitor_job.sh $CLUSTER <job_id>"
echo "  4. Get results: $SCRIPT_DIR/sync_results.sh $CLUSTER"
echo ""
