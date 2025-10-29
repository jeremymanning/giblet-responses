#!/bin/bash
# Utility functions for cluster management
# Source this script in other cluster scripts

# Colors for output
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export RED='\033[0;31m'
export BLUE='\033[0;34m'
export CYAN='\033[0;36m'
export NC='\033[0m'

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

# Function to read JSON value using Python
read_json() {
    local json_file=$1
    local key=$2
    python3 -c "import json; data=json.load(open('$json_file')); print(data.get('$key', ''))" 2>/dev/null
}

# Function to read JSON value using jq (fallback)
read_json_jq() {
    local json_file=$1
    local key=$2
    if command -v jq &> /dev/null; then
        jq -r ".${key}" "$json_file" 2>/dev/null
    else
        read_json "$json_file" "$key"
    fi
}

# Function to validate cluster name
validate_cluster() {
    local cluster=$1
    if [[ "$cluster" != "tensor01" && "$cluster" != "tensor02" ]]; then
        print_error "Invalid cluster name: $cluster. Must be 'tensor01' or 'tensor02'"
        return 1
    fi
    return 0
}

# Function to test SSH connection
test_ssh_connection() {
    local username=$1
    local password=$2
    local server=$3

    if ! command -v sshpass &> /dev/null; then
        print_warning "sshpass not installed"
        return 1
    fi

    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "$username@$server" "echo 'SSH test successful'" 2>/dev/null
}

# Function to execute remote command
run_remote_command() {
    local username=$1
    local password=$2
    local server=$3
    local command=$4

    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass not installed"
        return 1
    fi

    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "$username@$server" "$command" 2>/dev/null
}

# Function to load credentials
load_credentials() {
    local creds_file=$1
    local cluster=$2

    if [ ! -f "$creds_file" ]; then
        print_error "Credentials file not found: $creds_file"
        return 1
    fi

    # Source credentials into variables
    export CLUSTER_USERNAME=$(read_json "$creds_file" "username")
    export CLUSTER_PASSWORD=$(read_json "$creds_file" "password")
    export CLUSTER_SERVER=$(read_json "$creds_file" "server")
    export CLUSTER_BASE_PATH=$(read_json "$creds_file" "base_path")
    export CLUSTER_GPUS=$(read_json "$creds_file" "gpus")
    export CLUSTER_GPU_TYPE=$(read_json "$creds_file" "gpu_type")

    # Validate
    if [ -z "$CLUSTER_USERNAME" ] || [ -z "$CLUSTER_PASSWORD" ] || [ -z "$CLUSTER_SERVER" ]; then
        print_error "Failed to load credentials from $creds_file"
        return 1
    fi

    return 0
}

# Function to get timestamp
get_timestamp() {
    date +%Y%m%d_%H%M%S
}

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to ensure directory exists
ensure_directory() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    fi
}

# Function to get file size in human readable format
human_readable_size() {
    du -sh "$1" | cut -f1
}

# Export all functions
export -f print_info
export -f print_success
export -f print_warning
export -f print_error
export -f read_json
export -f read_json_jq
export -f validate_cluster
export -f test_ssh_connection
export -f run_remote_command
export -f load_credentials
export -f get_timestamp
export -f command_exists
export -f ensure_directory
export -f human_readable_size
