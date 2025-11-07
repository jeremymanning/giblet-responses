#!/bin/bash
#
# Shared Cluster Setup Function
# =============================
# Idempotent cluster environment setup that ALL cluster scripts should use
#
# This script provides a single setup_cluster_environment() function that:
#   1. Clones/updates the giblet-responses repository
#   2. Downloads the Sherlock dataset if needed
#   3. Creates the conda environment (giblet-py311)
#   4. Installs Python dependencies
#   5. Handles errors gracefully with clear messages
#
# Usage (from other scripts):
#   source "$(dirname "$0")/setup_cluster.sh"
#   setup_cluster_environment
#
# This script is designed to be IDEMPOTENT - safe to run multiple times
# without side effects. Each step checks if work is needed before acting.
#
# Created: 2025-11-06

# Exit on error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/ContextLab/giblet-responses.git"
REPO_DIR="$HOME/giblet-responses"
CONDA_ENV_NAME="giblet-py311"
PYTHON_VERSION="3.11"
MIN_DATA_FILES=15  # Minimum number of .nii.gz files expected

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

# ============================================================================
# STEP 1: CLONE/UPDATE REPOSITORY
# ============================================================================

setup_repository() {
    print_section "Step 1: Repository Setup"

    if [[ -d "$REPO_DIR" ]]; then
        print_info "Repository exists at $REPO_DIR"

        # Check if it's a valid git repository
        if [[ -d "$REPO_DIR/.git" ]]; then
            print_info "Updating repository from GitHub..."
            cd "$REPO_DIR"

            # Stash any local changes to avoid conflicts
            if ! git diff --quiet || ! git diff --cached --quiet; then
                print_warning "Local changes detected, stashing..."
                git stash push -m "Auto-stash before cluster setup $(date +%Y%m%d_%H%M%S)" || {
                    print_error "Failed to stash local changes"
                    return 1
                }
            fi

            # Pull latest changes
            if git pull origin main; then
                print_success "Repository updated successfully"
            else
                print_error "Failed to update repository"
                return 1
            fi
        else
            print_warning "$REPO_DIR exists but is not a git repository"
            print_error "Please remove $REPO_DIR and run this script again"
            return 1
        fi
    else
        print_info "Cloning repository from $REPO_URL"

        # Create parent directory if needed
        mkdir -p "$(dirname "$REPO_DIR")"

        if git clone "$REPO_URL" "$REPO_DIR"; then
            print_success "Repository cloned successfully to $REPO_DIR"
        else
            print_error "Failed to clone repository"
            print_info "You may need to set up SSH keys or check network connectivity"
            return 1
        fi
    fi

    # Verify repository directory exists and is accessible
    if [[ ! -d "$REPO_DIR" ]]; then
        print_error "Repository directory does not exist: $REPO_DIR"
        return 1
    fi

    cd "$REPO_DIR"
    print_success "Repository ready at $REPO_DIR"
}

# ============================================================================
# STEP 2: DOWNLOAD SHERLOCK DATASET
# ============================================================================

setup_dataset() {
    print_section "Step 2: Dataset Setup"

    cd "$REPO_DIR"

    DATA_DIR="$REPO_DIR/data/sherlock_nii"

    # Check if dataset already exists and has files
    if [[ -d "$DATA_DIR" ]]; then
        NII_COUNT=$(find "$DATA_DIR" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')

        if [[ $NII_COUNT -ge $MIN_DATA_FILES ]]; then
            print_success "Dataset already exists ($NII_COUNT .nii.gz files found)"
            print_info "Skipping download (dataset appears complete)"
            return 0
        else
            print_warning "Dataset directory exists but only has $NII_COUNT files (expected >= $MIN_DATA_FILES)"
            print_info "Will attempt to download dataset..."
        fi
    else
        print_info "Dataset not found at $DATA_DIR"
        print_info "Will download dataset..."
    fi

    # Check if download script exists
    DOWNLOAD_SCRIPT="$REPO_DIR/download_data_from_dropbox.sh"
    if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
        print_error "Download script not found: $DOWNLOAD_SCRIPT"
        print_warning "You may need to download the dataset manually"
        return 1
    fi

    # Make script executable
    chmod +x "$DOWNLOAD_SCRIPT"

    # Download dataset
    print_info "Starting dataset download (this may take several minutes for ~11GB)..."
    print_warning "Please be patient, this is downloading from Dropbox..."

    if bash "$DOWNLOAD_SCRIPT"; then
        # Verify download was successful
        NII_COUNT=$(find "$DATA_DIR" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [[ $NII_COUNT -ge $MIN_DATA_FILES ]]; then
            print_success "Dataset downloaded successfully ($NII_COUNT .nii.gz files)"
        else
            print_warning "Download completed but fewer files than expected ($NII_COUNT found)"
            print_warning "Training may fail if dataset is incomplete"
        fi
    else
        print_error "Dataset download failed"
        print_warning "You can try downloading manually by running:"
        print_warning "  cd $REPO_DIR && ./download_data_from_dropbox.sh"
        return 1
    fi
}

# ============================================================================
# STEP 3: SETUP CONDA ENVIRONMENT
# ============================================================================

setup_conda_environment() {
    print_section "Step 3: Conda Environment Setup"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found"
        print_info "Please install Miniconda or Anaconda first:"
        print_info "  https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi

    CONDA_VERSION=$(conda --version)
    print_success "Found conda: $CONDA_VERSION"

    # Initialize conda for this shell session
    eval "$(conda shell.bash hook)" || {
        # Try alternative initialization if first method fails
        source "$(conda info --base)/etc/profile.d/conda.sh" || {
            print_error "Failed to initialize conda"
            return 1
        }
    }

    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_success "Environment '$CONDA_ENV_NAME' already exists"
        print_info "Skipping environment creation (already exists)"
    else
        print_info "Creating new conda environment: $CONDA_ENV_NAME (Python $PYTHON_VERSION)"
        print_warning "This may take a few minutes..."

        if conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y; then
            print_success "Environment '$CONDA_ENV_NAME' created successfully"
        else
            print_error "Failed to create conda environment"
            return 1
        fi
    fi

    # Activate the environment
    print_info "Activating environment: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME" || {
        print_error "Failed to activate conda environment"
        return 1
    }

    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]]; then
        print_error "Environment activation failed (current: $CONDA_DEFAULT_ENV, expected: $CONDA_ENV_NAME)"
        return 1
    fi

    print_success "Environment activated: $CONDA_DEFAULT_ENV"
    print_info "Python version: $(python --version)"
}

# ============================================================================
# STEP 4: INSTALL DEPENDENCIES
# ============================================================================

install_dependencies() {
    print_section "Step 4: Installing Dependencies"

    cd "$REPO_DIR"

    # Ensure environment is activated
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]]; then
        print_info "Re-activating conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME" || {
            print_error "Failed to activate conda environment"
            return 1
        }
    fi

    # Check which requirements file to use (prefer requirements_conda.txt)
    REQUIREMENTS_FILE=""
    if [[ -f "$REPO_DIR/requirements_conda.txt" ]]; then
        REQUIREMENTS_FILE="$REPO_DIR/requirements_conda.txt"
        print_info "Using requirements_conda.txt"
    elif [[ -f "$REPO_DIR/requirements.txt" ]]; then
        REQUIREMENTS_FILE="$REPO_DIR/requirements.txt"
        print_info "Using requirements.txt"
    else
        print_error "No requirements file found"
        print_info "Looked for: requirements_conda.txt or requirements.txt"
        return 1
    fi

    # Check if packages are already installed by testing key imports
    print_info "Checking if dependencies are already installed..."
    if python -c "import torch; import sentence_transformers; import nibabel" 2>/dev/null; then
        print_success "Key dependencies already installed"
        print_info "Checking for updates..."

        # Only reinstall if requirements file is newer than last install
        # (We'll skip this check for simplicity and just do a quick install)
        print_info "Running pip install to ensure all dependencies are current..."
    else
        print_info "Dependencies not installed or incomplete"
        print_info "This will take several minutes (PyTorch and ML libraries)..."
    fi

    # Upgrade pip first
    print_info "Upgrading pip..."
    pip install --upgrade pip -q || {
        print_warning "Failed to upgrade pip (continuing anyway)"
    }

    # Install ffmpeg if not present (needed for audio processing)
    if ! python -c "import av" 2>/dev/null; then
        print_info "Installing ffmpeg for audio processing..."
        conda install -y -c conda-forge ffmpeg -q || {
            print_warning "Failed to install ffmpeg from conda (may already be installed)"
        }

        print_info "Installing PyAV (Python bindings for ffmpeg)..."
        pip install av -q || {
            print_warning "Failed to install PyAV (continuing anyway)"
        }
    fi

    # Install dependencies from requirements file
    print_info "Installing packages from $REQUIREMENTS_FILE..."
    if pip install -r "$REQUIREMENTS_FILE" --no-cache-dir; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install some dependencies"
        print_warning "Training may fail if dependencies are incomplete"
        return 1
    fi

    # Special handling for tf-keras if needed
    if grep -q "tensorflow" "$REQUIREMENTS_FILE" 2>/dev/null; then
        if ! python -c "import keras" 2>/dev/null; then
            print_info "Installing tf-keras (Keras 3 compatibility)..."
            pip install tf-keras -q || {
                print_warning "Failed to install tf-keras (may not be needed)"
            }
        fi
    fi

    # Verify key packages
    print_info "Verifying installation..."

    # Check PyTorch
    if python -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        print_success "PyTorch $TORCH_VERSION installed"

        # Check CUDA availability
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.version.cuda else 'N/A')")
            GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
            print_success "CUDA available: version $CUDA_VERSION ($GPU_COUNT GPUs detected)"
        else
            print_warning "CUDA not available (CPU only mode)"
            print_warning "Training will be slow without GPU acceleration"
        fi
    else
        print_error "PyTorch not installed correctly"
        return 1
    fi

    # Check other key packages
    PACKAGES=("sentence_transformers" "nibabel" "numpy" "pandas")
    for pkg in "${PACKAGES[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            print_success "$pkg ✓"
        else
            print_error "$pkg not installed"
            return 1
        fi
    done

    print_success "All dependencies verified"
}

# ============================================================================
# MAIN SETUP FUNCTION (exported for use by other scripts)
# ============================================================================

setup_cluster_environment() {
    print_section "Giblet Cluster Environment Setup"
    print_info "This will set up the complete environment for training"
    print_info "All steps are idempotent - safe to run multiple times"
    echo ""

    # Step 1: Repository
    if ! setup_repository; then
        print_error "Repository setup failed"
        return 1
    fi

    # Step 2: Dataset
    if ! setup_dataset; then
        print_warning "Dataset setup failed (continuing anyway)"
        print_warning "You may need to download the dataset manually"
    fi

    # Step 3: Conda environment
    if ! setup_conda_environment; then
        print_error "Conda environment setup failed"
        return 1
    fi

    # Step 4: Dependencies
    if ! install_dependencies; then
        print_error "Dependency installation failed"
        return 1
    fi

    # Success!
    print_section "Setup Complete!"
    print_success "Environment is ready for training"
    echo ""
    print_info "Repository:   $REPO_DIR"
    print_info "Environment:  $CONDA_ENV_NAME"
    print_info "Python:       $(python --version 2>&1)"
    echo ""
    print_success "You can now run training scripts"
    echo ""
}

# ============================================================================
# SCRIPT CAN ALSO BE RUN DIRECTLY (not just sourced)
# ============================================================================

# If script is run directly (not sourced), execute setup
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_info "Running setup directly..."
    setup_cluster_environment || {
        print_error "Setup failed"
        exit 1
    }

    print_section "Next Steps"
    echo -e "  1. Activate the environment:"
    echo -e "     ${CYAN}conda activate $CONDA_ENV_NAME${NC}"
    echo ""
    echo -e "  2. Navigate to repository:"
    echo -e "     ${CYAN}cd $REPO_DIR${NC}"
    echo ""
    echo -e "  3. Run training:"
    echo -e "     ${CYAN}./run_giblet.sh --task train${NC}"
    echo ""
fi
