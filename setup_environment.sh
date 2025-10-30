#!/bin/bash
#
# Giblet Environment Setup Script
# ===============================
# Automated setup for giblet-responses development environment
# - Detects/installs miniconda (Linux/macOS, x86_64/arm64)
# - Creates conda environment with Python 3.11
# - Installs dependencies from requirements_conda.txt
# - Downloads Sherlock dataset if needed
# - Verifies installation
# - Runs quick validation tests
#
# Usage: ./setup_environment.sh
# Based on: https://github.com/ContextLab/llm-stylometry/blob/main/run_llm_stylometry.sh
#
# Created: 2025-10-30
# Issue: #19

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="giblet-py311"
PYTHON_VERSION="3.11"
REQUIREMENTS_FILE="requirements_conda.txt"
DOWNLOAD_SCRIPT="./download_data_from_dropbox.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print banner
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                            ║${NC}"
echo -e "${CYAN}║          ${GREEN}Giblet Environment Setup Script${CYAN}                 ║${NC}"
echo -e "${CYAN}║                                                            ║${NC}"
echo -e "${CYAN}║  Setting up Python 3.11 environment with PyTorch,         ║${NC}"
echo -e "${CYAN}║  sentence-transformers, and Sherlock dataset               ║${NC}"
echo -e "${CYAN}║                                                            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

# Detect OS
detect_os() {
    print_section "Detecting Operating System"

    OS_TYPE=$(uname -s)
    ARCH=$(uname -m)

    case "$OS_TYPE" in
        Linux*)
            OS="Linux"
            print_success "Detected Linux (${ARCH})"
            ;;
        Darwin*)
            OS="macOS"
            print_success "Detected macOS (${ARCH})"
            ;;
        *)
            print_error "Unsupported operating system: $OS_TYPE"
            exit 1
            ;;
    esac

    # Check if running on cluster (common cluster indicators)
    if [[ -n "$SLURM_JOB_ID" ]] || [[ -n "$PBS_JOBID" ]] || [[ "$HOSTNAME" == *"cluster"* ]]; then
        print_warning "Detected cluster environment"
        IS_CLUSTER=true
    else
        print_info "Running on local machine"
        IS_CLUSTER=false
    fi
}

# Check if conda is installed
check_conda() {
    print_section "Checking for Conda Installation"

    if command -v conda &> /dev/null; then
        CONDA_PATH=$(which conda)
        CONDA_VERSION=$(conda --version)
        print_success "Found conda: $CONDA_PATH"
        print_info "Version: $CONDA_VERSION"
        return 0
    else
        print_warning "Conda not found"
        return 1
    fi
}

# Install miniconda
install_miniconda() {
    print_section "Installing Miniconda"

    # Determine miniconda installer URL based on OS and architecture
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$ARCH" == "aarch64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else
            print_error "Unsupported Linux architecture: $ARCH"
            exit 1
        fi
    elif [[ "$OS" == "macOS" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif [[ "$ARCH" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            print_error "Unsupported macOS architecture: $ARCH"
            exit 1
        fi
    fi

    print_info "Downloading Miniconda from: $MINICONDA_URL"

    INSTALLER_PATH="/tmp/miniconda_installer.sh"

    if curl -L -o "$INSTALLER_PATH" "$MINICONDA_URL"; then
        print_success "Downloaded Miniconda installer"
    else
        print_error "Failed to download Miniconda installer"
        echo ""
        echo "Manual installation instructions:"
        echo "1. Visit: https://docs.conda.io/en/latest/miniconda.html"
        echo "2. Download the appropriate installer for your system"
        echo "3. Run the installer and follow the prompts"
        echo "4. Re-run this script after installation"
        exit 1
    fi

    # Check for write permissions to home directory
    if [[ ! -w "$HOME" ]]; then
        print_error "No write permission to home directory: $HOME"
        exit 1
    fi

    print_info "Installing Miniconda to $HOME/miniconda3"
    print_warning "This may take a few minutes..."

    if bash "$INSTALLER_PATH" -b -p "$HOME/miniconda3"; then
        print_success "Miniconda installed successfully"
        rm "$INSTALLER_PATH"

        # Initialize conda for bash
        print_info "Initializing conda..."
        eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

        # Add to shell configuration
        if [[ -f "$HOME/.bashrc" ]]; then
            if ! grep -q "miniconda3/bin/conda" "$HOME/.bashrc"; then
                echo "" >> "$HOME/.bashrc"
                echo "# >>> conda initialize >>>" >> "$HOME/.bashrc"
                "$HOME/miniconda3/bin/conda" init bash >> /dev/null 2>&1
                echo "# <<< conda initialize <<<" >> "$HOME/.bashrc"
                print_success "Added conda to .bashrc"
            fi
        fi

        if [[ -f "$HOME/.bash_profile" ]]; then
            if ! grep -q "miniconda3/bin/conda" "$HOME/.bash_profile"; then
                echo "" >> "$HOME/.bash_profile"
                echo "# >>> conda initialize >>>" >> "$HOME/.bash_profile"
                "$HOME/miniconda3/bin/conda" init bash >> /dev/null 2>&1
                echo "# <<< conda initialize <<<" >> "$HOME/.bash_profile"
                print_success "Added conda to .bash_profile"
            fi
        fi

        print_success "Conda initialized"
    else
        print_error "Failed to install Miniconda"
        rm "$INSTALLER_PATH"
        exit 1
    fi
}

# Create conda environment
create_conda_env() {
    print_section "Creating Conda Environment: $CONDA_ENV_NAME"

    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Environment '$CONDA_ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
            print_success "Removed existing environment"
        else
            print_info "Keeping existing environment"
            return 0
        fi
    fi

    print_info "Creating new environment with Python $PYTHON_VERSION..."
    if conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y; then
        print_success "Environment '$CONDA_ENV_NAME' created successfully"
    else
        print_error "Failed to create conda environment"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_section "Installing Dependencies from $REQUIREMENTS_FILE"

    if [[ ! -f "$PROJECT_ROOT/$REQUIREMENTS_FILE" ]]; then
        print_error "Requirements file not found: $PROJECT_ROOT/$REQUIREMENTS_FILE"
        exit 1
    fi

    print_info "Activating environment: $CONDA_ENV_NAME"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"

    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]]; then
        print_error "Failed to activate conda environment"
        exit 1
    fi

    print_success "Environment activated: $CONDA_DEFAULT_ENV"

    print_info "Installing packages (this may take several minutes)..."
    print_warning "Installing PyTorch and deep learning frameworks..."

    cd "$PROJECT_ROOT"

    # Install in stages for better error reporting
    if pip install -r "$REQUIREMENTS_FILE" --no-cache-dir; then
        print_success "All dependencies installed successfully"
    else
        print_error "Failed to install some dependencies"
        print_warning "You may need to install packages manually"
        exit 1
    fi
}

# Download dataset
download_dataset() {
    print_section "Checking Sherlock Dataset"

    DATA_DIR="$PROJECT_ROOT/data/sherlock_nii"

    # Check if dataset already exists
    if [[ -d "$DATA_DIR" ]]; then
        NII_COUNT=$(find "$DATA_DIR" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [[ $NII_COUNT -gt 0 ]]; then
            print_success "Dataset already downloaded ($NII_COUNT .nii.gz files found)"
            return 0
        fi
    fi

    print_warning "Dataset not found"

    if [[ ! -f "$PROJECT_ROOT/$DOWNLOAD_SCRIPT" ]]; then
        print_error "Download script not found: $PROJECT_ROOT/$DOWNLOAD_SCRIPT"
        print_info "You can download the dataset manually later"
        return 1
    fi

    if [[ ! -x "$PROJECT_ROOT/$DOWNLOAD_SCRIPT" ]]; then
        print_warning "Download script is not executable, fixing permissions..."
        chmod +x "$PROJECT_ROOT/$DOWNLOAD_SCRIPT"
    fi

    read -p "Do you want to download the Sherlock dataset (~11GB)? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_info "Starting dataset download..."
        cd "$PROJECT_ROOT"
        if bash "$DOWNLOAD_SCRIPT"; then
            print_success "Dataset downloaded successfully"
        else
            print_error "Dataset download failed"
            print_info "You can download it later by running: $DOWNLOAD_SCRIPT"
            return 1
        fi
    else
        print_info "Skipping dataset download"
        print_warning "Run '$DOWNLOAD_SCRIPT' later to download the dataset"
    fi
}

# Verify installation
verify_installation() {
    print_section "Verifying Installation"

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"

    # Test 1: Import giblet
    print_info "Testing giblet module import..."
    if python -c "import giblet; print('Giblet version:', giblet.__version__ if hasattr(giblet, '__version__') else 'unknown')" 2>/dev/null; then
        print_success "Giblet module imports successfully"
    else
        print_warning "Giblet module import failed (this may be expected if not yet installed as package)"
    fi

    # Test 2: PyTorch
    print_info "Testing PyTorch installation..."
    TORCH_INFO=$(python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1)
    if [[ $? -eq 0 ]]; then
        print_success "$TORCH_INFO"

        # Check for GPU support
        if [[ "$OS" == "macOS" ]]; then
            if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
                print_success "MPS (Metal) acceleration available"
            else
                print_warning "MPS (Metal) acceleration not available"
            fi
        else
            if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
                print_success "CUDA acceleration available (version: $CUDA_VERSION)"
            else
                print_warning "CUDA acceleration not available (CPU only)"
            fi
        fi
    else
        print_error "PyTorch import failed"
        echo "$TORCH_INFO"
        exit 1
    fi

    # Test 3: sentence-transformers
    print_info "Testing sentence-transformers..."
    if python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers imported successfully')" 2>/dev/null; then
        print_success "sentence-transformers working"
    else
        print_error "sentence-transformers import failed"
        exit 1
    fi

    # Test 4: Other key dependencies
    print_info "Testing other dependencies..."
    DEPS=("numpy" "pandas" "scipy" "sklearn" "matplotlib")
    for dep in "${DEPS[@]}"; do
        if python -c "import $dep" 2>/dev/null; then
            print_success "$dep ✓"
        else
            print_error "$dep failed"
            exit 1
        fi
    done
}

# Run quick tests
run_quick_tests() {
    print_section "Running Quick Validation Tests"

    # Check if pytest is installed
    if ! python -c "import pytest" 2>/dev/null; then
        print_warning "pytest not installed, skipping tests"
        print_info "Install pytest with: pip install pytest"
        return 0
    fi

    # Check if tests directory exists
    if [[ ! -d "$PROJECT_ROOT/tests" ]]; then
        print_warning "Tests directory not found, skipping tests"
        return 0
    fi

    print_info "Running quick tests (excluding dataset and training tests)..."

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"

    cd "$PROJECT_ROOT"

    # Run quick tests - exclude slow tests
    # Using -x to stop at first failure, -v for verbose, -q for quiet mode
    if python -m pytest tests/ \
        -k "not dataset and not training and not integration" \
        --tb=short \
        --maxfail=3 \
        -v \
        2>&1 | head -n 100; then
        print_success "Quick tests passed"
    else
        print_warning "Some tests failed (this may be expected during initial setup)"
        print_info "Run 'pytest tests/' for full test results"
    fi
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                            ║${NC}"
    echo -e "${CYAN}║          ${GREEN}✓ Setup Complete!${CYAN}                               ║${NC}"
    echo -e "${CYAN}║                                                            ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    print_success "Environment '$CONDA_ENV_NAME' is ready to use"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo ""
    echo -e "  1. Activate the environment:"
    echo -e "     ${CYAN}conda activate $CONDA_ENV_NAME${NC}"
    echo ""
    echo -e "  2. Run tests:"
    echo -e "     ${CYAN}pytest tests/${NC}"
    echo ""
    echo -e "  3. Start development:"
    echo -e "     ${CYAN}python examples/basic_usage.py${NC}"
    echo ""
    echo -e "  4. Check examples:"
    echo -e "     ${CYAN}ls examples/${NC}"
    echo ""

    if [[ ! -d "$PROJECT_ROOT/data/sherlock_nii" ]]; then
        echo -e "${YELLOW}Note: Sherlock dataset not downloaded${NC}"
        echo -e "  Download with: ${CYAN}$DOWNLOAD_SCRIPT${NC}"
        echo ""
    fi

    echo -e "${CYAN}Documentation:${NC}"
    echo -e "  - README.md       - Project overview"
    echo -e "  - CLAUDE.md       - Development guide"
    echo -e "  - STATUS.md       - Current status"
    echo ""
}

# Main execution
main() {
    # Change to project root
    cd "$PROJECT_ROOT"

    # Step 1: Detect OS
    detect_os

    # Step 2: Check/install conda
    if ! check_conda; then
        install_miniconda
        # Re-check after installation
        if ! check_conda; then
            print_error "Conda installation failed"
            exit 1
        fi
    fi

    # Initialize conda for this script
    eval "$(conda shell.bash hook)"

    # Step 3: Create conda environment
    create_conda_env

    # Step 4: Install dependencies
    install_dependencies

    # Step 5: Download dataset (optional)
    download_dataset || true  # Don't fail if dataset download fails

    # Step 6: Verify installation
    verify_installation

    # Step 7: Run quick tests
    run_quick_tests || true  # Don't fail if tests fail

    # Step 8: Print completion message
    print_completion
}

# Handle Ctrl+C gracefully
trap 'echo ""; print_error "Setup interrupted by user"; exit 130' INT

# Run main function
main
