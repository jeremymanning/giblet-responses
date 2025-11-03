#!/bin/bash
#
# run_giblet.sh - Universal execution script for GIBLET project
#
# Purpose: Provides unified interface for training, testing, and validation
#          across local (macOS) and cluster (Linux) environments
#
# Requirements:
#   - Conda environment: giblet-py311
#   - PyTorch with GPU support (CUDA or MPS)
#   - Data files in data/ directory
#
# Usage:
#   ./run_giblet.sh --task train --config CONFIG_PATH --gpus NUM_GPUS
#   ./run_giblet.sh --task test
#   ./run_giblet.sh --task validate_data
#   ./run_giblet.sh --help
#

set -e  # Exit on error

# ============================================================================
# Color codes for output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Helper functions
# ============================================================================

print_header() {
    echo -e "\n${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_step() {
    echo -e "${BOLD}→${NC} $1"
}

# ============================================================================
# Environment detection
# ============================================================================

detect_environment() {
    print_header "Environment Detection"

    # Detect hostname
    HOSTNAME=$(hostname)
    print_info "Hostname: $HOSTNAME"

    # Check if running on cluster
    if [[ "$HOSTNAME" == *"tensor01"* ]] || [[ "$HOSTNAME" == *"tensor02"* ]]; then
        ENV_TYPE="cluster"
        print_success "Running on cluster: $HOSTNAME"
    else
        ENV_TYPE="local"
        print_success "Running locally"
    fi

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        print_info "OS: macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        print_info "OS: Linux"
    else
        OS_TYPE="unknown"
        print_warning "OS: Unknown ($OSTYPE)"
    fi

    # Check if in screen session
    if [[ -n "$STY" ]]; then
        IN_SCREEN=true
        print_success "Running in screen session: $STY"
    else
        IN_SCREEN=false
        print_info "Not in screen session"
        if [[ "$ENV_TYPE" == "cluster" ]]; then
            print_warning "Recommended: Run long training jobs in screen"
        fi
    fi
}

# ============================================================================
# Conda environment verification
# ============================================================================

verify_conda_env() {
    print_header "Conda Environment Verification"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Anaconda or Miniconda."
        return 1
    fi
    print_success "Conda is available"

    # Check if giblet-py311 environment exists
    if ! conda env list | grep -q "giblet-py311"; then
        print_error "Conda environment 'giblet-py311' not found"
        print_info "Create it with: conda env create -f environment.yml"
        print_info "Or manually: conda create -n giblet-py311 python=3.11"
        return 1
    fi
    print_success "Environment 'giblet-py311' exists"

    # Check if environment is activated
    CURRENT_ENV=$(conda info --envs | grep '\*' | awk '{print $1}')
    if [[ "$CURRENT_ENV" != "giblet-py311" ]]; then
        print_warning "Environment 'giblet-py311' not activated"
        print_info "Activating now..."

        # Source conda and activate
        CONDA_BASE=$(conda info --base)
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate giblet-py311

        if [[ $? -eq 0 ]]; then
            print_success "Activated giblet-py311"
        else
            print_error "Failed to activate giblet-py311"
            return 1
        fi
    else
        print_success "Environment 'giblet-py311' is active"
    fi

    # Verify Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python version: $PYTHON_VERSION"

    return 0
}

# ============================================================================
# GPU detection
# ============================================================================

detect_gpus() {
    print_header "GPU Detection"

    NUM_AVAILABLE_GPUS=0
    GPU_TYPE="none"

    # Check for NVIDIA GPUs (CUDA)
    if command -v nvidia-smi &> /dev/null; then
        NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)
        GPU_TYPE="cuda"
        print_success "NVIDIA GPUs detected: $NUM_AVAILABLE_GPUS"
        echo ""
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
        echo ""
    # Check for Apple Silicon (MPS)
    elif [[ "$OS_TYPE" == "macos" ]]; then
        # Check if MPS is available via Python
        if python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
            NUM_AVAILABLE_GPUS=1
            GPU_TYPE="mps"
            print_success "Apple Silicon GPU (MPS) detected"
        else
            print_warning "MPS not available"
        fi
    else
        print_warning "No GPU detected. Will use CPU only."
    fi

    if [[ $NUM_AVAILABLE_GPUS -eq 0 ]]; then
        print_info "Training will run on CPU (slow)"
    fi

    return 0
}

# ============================================================================
# Data verification
# ============================================================================

verify_data() {
    print_header "Data Verification"

    # Check if data directory exists
    if [[ ! -d "data" ]]; then
        print_error "Data directory not found"
        print_info "Run: ./download_data_from_dropbox.sh"
        return 1
    fi
    print_success "Data directory exists"

    # Check for key data files
    local all_found=true

    # Check stimulus video
    if [[ -f "data/stimuli_Sherlock.m4v" ]]; then
        print_success "Stimulus video found"
    else
        print_error "Stimulus video not found: data/stimuli_Sherlock.m4v"
        all_found=false
    fi

    # Check for fMRI data
    if [[ -d "data/sherlock_nii" ]] && [[ $(ls -1 data/sherlock_nii/*.nii.gz 2>/dev/null | wc -l) -gt 0 ]]; then
        NUM_SUBJECTS=$(ls -1 data/sherlock_nii/*.nii.gz 2>/dev/null | wc -l)
        print_success "fMRI data found: $NUM_SUBJECTS subject files"
    else
        print_error "fMRI data not found in data/sherlock_nii/"
        all_found=false
    fi

    # Check annotations
    if [[ -f "data/annotations.xlsx" ]]; then
        print_success "Annotations found"
    else
        print_warning "Annotations not found: data/annotations.xlsx (optional)"
    fi

    if [[ "$all_found" == false ]]; then
        print_error "Missing required data files"
        print_info "Download with: ./download_data_from_dropbox.sh"
        return 1
    fi

    print_success "All required data files present"
    return 0
}

# ============================================================================
# Task: Train
# ============================================================================

task_train() {
    local config_file=$1
    local num_gpus=$2

    print_header "Training Task"

    # Verify config file exists
    if [[ -z "$config_file" ]]; then
        print_error "Config file not specified"
        print_info "Use: --config path/to/config.yaml"
        return 1
    fi

    if [[ ! -f "$config_file" ]]; then
        print_error "Config file not found: $config_file"
        return 1
    fi
    print_success "Config file: $config_file"

    # Determine number of GPUs to use
    if [[ -z "$num_gpus" ]]; then
        if [[ $NUM_AVAILABLE_GPUS -gt 0 ]]; then
            num_gpus=1
            print_info "No --gpus specified, using 1 GPU"
        else
            num_gpus=0
            print_warning "No GPUs available, using CPU"
        fi
    fi

    # Validate GPU request
    if [[ $num_gpus -gt $NUM_AVAILABLE_GPUS ]]; then
        print_error "Requested $num_gpus GPUs but only $NUM_AVAILABLE_GPUS available"
        return 1
    fi

    print_info "Using $num_gpus GPU(s)"

    # Setup training command
    if [[ $num_gpus -eq 0 ]]; then
        # CPU only
        print_warning "Training on CPU (very slow)"
        TRAIN_CMD="python scripts/train.py --config $config_file"

    elif [[ $num_gpus -eq 1 ]]; then
        # Single GPU
        print_success "Single GPU training"
        if [[ "$GPU_TYPE" == "mps" ]]; then
            # MPS (Apple Silicon) - no special setup needed
            TRAIN_CMD="python scripts/train.py --config $config_file"
        else
            # CUDA - set visible devices
            export CUDA_VISIBLE_DEVICES=0
            TRAIN_CMD="python scripts/train.py --config $config_file"
        fi

    else
        # Multi-GPU distributed training
        print_success "Distributed training on $num_gpus GPUs"

        if [[ "$GPU_TYPE" == "cuda" ]]; then
            # Set GPU devices (0,1,2,... up to num_gpus-1)
            GPU_IDS=$(seq -s, 0 $((num_gpus-1)))
            export CUDA_VISIBLE_DEVICES=$GPU_IDS
            print_info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

            # Use torchrun for distributed training
            TRAIN_CMD="torchrun --nproc_per_node=$num_gpus scripts/train.py --config $config_file --distributed"
        else
            print_error "Multi-GPU training only supported with CUDA"
            return 1
        fi
    fi

    # Print command
    print_step "Executing: $TRAIN_CMD"
    echo ""

    # Execute training
    eval $TRAIN_CMD

    if [[ $? -eq 0 ]]; then
        print_success "Training completed successfully"
        return 0
    else
        print_error "Training failed"
        return 1
    fi
}

# ============================================================================
# Task: Test
# ============================================================================

task_test() {
    print_header "Running Test Suite"

    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        print_error "pytest not found"
        print_info "Install with: pip install pytest"
        return 1
    fi

    print_step "Running pytest..."
    echo ""

    # Run pytest with verbose output
    pytest tests/ -v --tb=short

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        print_success "All tests passed"
        return 0
    else
        print_error "Some tests failed"
        return $exit_code
    fi
}

# ============================================================================
# Task: Validate Data
# ============================================================================

task_validate_data() {
    print_header "Data Validation"

    # Check if validation script exists
    if [[ ! -f "validate_all_modalities.py" ]]; then
        print_error "Validation script not found: validate_all_modalities.py"
        return 1
    fi

    print_step "Running validation script..."
    echo ""

    python validate_all_modalities.py

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        print_success "Data validation completed"
        return 0
    else
        print_error "Data validation failed"
        return $exit_code
    fi
}

# ============================================================================
# Usage/Help
# ============================================================================

show_usage() {
    cat << EOF
${BOLD}GIBLET - Multimodal Autoencoder Training Script${NC}

${BOLD}USAGE:${NC}
    ./run_giblet.sh --task TASK [OPTIONS]

${BOLD}TASKS:${NC}
    train           Train the multimodal autoencoder
    test            Run the test suite
    validate_data   Validate dataset integrity

${BOLD}OPTIONS:${NC}
    --config PATH   Path to training config YAML file (required for train)
    --gpus NUM      Number of GPUs to use (default: 1 if available, 0 otherwise)
    --help          Show this help message

${BOLD}EXAMPLES:${NC}
    ${GREEN}# Local training with 1 GPU${NC}
    ./run_giblet.sh --task train --config configs/training/test_config.yaml --gpus 1

    ${GREEN}# Distributed training with 8 GPUs${NC}
    ./run_giblet.sh --task train --config configs/cluster/cluster_train_config.yaml --gpus 8

    ${GREEN}# CPU training (no GPU)${NC}
    ./run_giblet.sh --task train --config configs/training/test_config.yaml --gpus 0

    ${GREEN}# Run tests${NC}
    ./run_giblet.sh --task test

    ${GREEN}# Validate data${NC}
    ./run_giblet.sh --task validate_data

${BOLD}ENVIRONMENT:${NC}
    - Requires conda environment: giblet-py311
    - Activate with: conda activate giblet-py311
    - Script will attempt to activate if not already active

${BOLD}DISTRIBUTED TRAINING:${NC}
    - Single GPU (--gpus 1): Direct python execution
    - Multi-GPU (--gpus >1): Uses torchrun with --distributed flag
    - Sets CUDA_VISIBLE_DEVICES automatically
    - Only CUDA GPUs support multi-GPU training

${BOLD}CLUSTER USAGE:${NC}
    - Recommended: Run in screen session for long jobs
    - Start screen: screen -S training
    - Detach: Ctrl+A then D
    - Reattach: screen -r training

${BOLD}FILES:${NC}
    scripts/train.py                      Training script
    configs/training/*.yaml               Training config files
    configs/cluster/cluster_train_config  Cluster training config
    validate_all_modalities.py            Data validation script
    tests/                                Test directory

${BOLD}MORE INFO:${NC}
    - See ENVIRONMENT_SETUP.md for setup instructions
    - See CLAUDE.md for project overview
    - GitHub: https://github.com/ContextLab/giblet-responses

EOF
}

# ============================================================================
# Main execution
# ============================================================================

main() {
    # Parse command line arguments
    TASK=""
    CONFIG_FILE=""
    NUM_GPUS=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --task)
                TASK="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done

    # Check if task is specified
    if [[ -z "$TASK" ]]; then
        print_error "No task specified"
        echo ""
        show_usage
        exit 1
    fi

    # Print banner
    echo ""
    print_header "GIBLET - Multimodal Autoencoder"

    # Detect environment
    detect_environment

    # Verify conda environment
    if ! verify_conda_env; then
        exit 1
    fi

    # Detect GPUs
    detect_gpus

    # Execute task
    case $TASK in
        train)
            # Verify data before training
            if ! verify_data; then
                exit 1
            fi

            # Run training
            if ! task_train "$CONFIG_FILE" "$NUM_GPUS"; then
                exit 1
            fi
            ;;

        test)
            if ! task_test; then
                exit 1
            fi
            ;;

        validate_data)
            if ! verify_data; then
                exit 1
            fi
            if ! task_validate_data; then
                exit 1
            fi
            ;;

        *)
            print_error "Unknown task: $TASK"
            print_info "Valid tasks: train, test, validate_data"
            exit 1
            ;;
    esac

    # Success
    echo ""
    print_header "Task Completed Successfully"
    exit 0
}

# Run main function
main "$@"
