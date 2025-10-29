#!/bin/bash
# Example workflow demonstrating cluster script usage
# This script shows a complete end-to-end example of:
# 1. Setting up cluster
# 2. Submitting multiple jobs
# 3. Monitoring progress
# 4. Syncing results

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_section() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "  $1"
    echo "==========================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ============================================================================
print_section "Cluster Workflow Example"
# ============================================================================

print_info "This example demonstrates the complete workflow:"
echo "  1. Setup cluster environment"
echo "  2. Submit training jobs"
echo "  3. Monitor job progress"
echo "  4. Sync results back to local machine"
echo ""

# ============================================================================
# STEP 1: Setup (optional - only needed once per cluster)
# ============================================================================

print_section "Step 1: Cluster Setup (Optional)"

print_warning "Cluster setup is usually a one-time operation"
print_info "To setup a cluster, run:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  ./setup_cluster.sh tensor01"
echo "  # or"
echo "  ./setup_cluster.sh tensor02"
echo ""

read -p "Setup cluster now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Which cluster? (tensor01/tensor02): " cluster
    if [[ "$cluster" == "tensor01" || "$cluster" == "tensor02" ]]; then
        print_info "Setting up $cluster..."
        bash "$SCRIPT_DIR/setup_cluster.sh" "$cluster"
        print_success "Setup complete!"
        sleep 5
    fi
fi

# ============================================================================
# STEP 2: Submit Jobs
# ============================================================================

print_section "Step 2: Submit Training Jobs"

# Example 1: Single job
print_info "Example 1: Submit a single job"
echo ""
echo "Command:"
echo "  ./submit_job.sh tensor01 my_job demo_decoder.py --epochs 100"
echo ""

read -p "Submit job on tensor01? (y/n) " -n 1 -r
echo

JOB_ID=""
CLUSTER="tensor01"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Submitting job..."
    echo ""

    # You can customize these parameters
    JOB_NAME="example_job_$(date +%s)"
    SCRIPT_PATH="demo_decoder.py"
    PYTHON_ARGS="--epochs 10"

    print_info "Job details:"
    echo "  Cluster: $CLUSTER"
    echo "  Job name: $JOB_NAME"
    echo "  Script: $SCRIPT_PATH"
    echo "  Args: $PYTHON_ARGS"
    echo ""

    # Run the actual command and capture output
    if OUTPUT=$(bash "$SCRIPT_DIR/submit_job.sh" "$CLUSTER" "$JOB_NAME" "$SCRIPT_PATH" $PYTHON_ARGS 2>&1); then
        echo "$OUTPUT"

        # Try to extract job ID
        JOB_ID=$(echo "$OUTPUT" | grep -oP '(?<=Job ID: )\d+' || echo "")

        if [ -n "$JOB_ID" ]; then
            print_success "Job submitted with ID: $JOB_ID"
        else
            print_warning "Could not extract job ID from output"
            JOB_ID=$(read -p "Enter job ID manually: ")
        fi
    else
        print_warning "Job submission encountered issues"
        print_info "Output: $OUTPUT"
    fi
fi

echo ""

# Example 2: Multiple jobs
print_info "Example 2: Submit multiple jobs (commented out)"
echo ""
echo "To submit multiple jobs, you can use:"
cat <<'EOF'
# Submit jobs in parallel
./submit_job.sh tensor01 exp_seed1 train.py --seed 1 &
./submit_job.sh tensor01 exp_seed2 train.py --seed 2 &
./submit_job.sh tensor02 exp_seed3 train.py --seed 3 &

# Wait for all submissions to complete
wait

# Or capture job IDs:
JOB_1=$(./submit_job.sh tensor01 exp1 train.py | grep -oP '(?<=Job ID: )\d+')
JOB_2=$(./submit_job.sh tensor01 exp2 train.py | grep -oP '(?<=Job ID: )\d+')
echo "Submitted jobs: $JOB_1, $JOB_2"
EOF
echo ""

# ============================================================================
# STEP 3: Monitor Job Progress
# ============================================================================

print_section "Step 3: Monitor Job Progress"

if [ -n "$JOB_ID" ]; then
    print_info "Monitoring job $JOB_ID on $CLUSTER"
    echo ""

    print_info "Command options:"
    echo "  ./monitor_job.sh $CLUSTER $JOB_ID              # Check status (shows last 50 lines)"
    echo "  ./monitor_job.sh $CLUSTER $JOB_ID --tail       # Follow log in real-time"
    echo "  ./monitor_job.sh $CLUSTER $JOB_ID --error      # Show error log"
    echo "  ./monitor_job.sh $CLUSTER $JOB_ID --error --tail # Follow error log"
    echo ""

    read -p "Monitor job status? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Getting job status..."
        echo ""
        bash "$SCRIPT_DIR/monitor_job.sh" "$CLUSTER" "$JOB_ID"
    fi
else
    print_warning "No job ID available - skipping monitoring"
    print_info "If you submitted a job, you can monitor it with:"
    echo "  ./monitor_job.sh <cluster> <job_id>"
fi

echo ""

# ============================================================================
# STEP 4: Continuous Monitoring Example
# ============================================================================

print_section "Step 4: Continuous Monitoring (Example)"

if [ -n "$JOB_ID" ]; then
    print_info "You can monitor job progress periodically with a loop:"
    echo ""
    cat <<EOF
# Check job status every 30 seconds for 5 minutes
for i in {1..10}; do
  echo "Check \$i at \$(date)..."
  ./monitor_job.sh $CLUSTER $JOB_ID --tail
  echo "---"
  sleep 30
done
EOF
    echo ""
fi

# ============================================================================
# STEP 5: Sync Results
# ============================================================================

print_section "Step 5: Retrieve Results"

print_info "Once the job completes, sync results with:"
echo "  ./sync_results.sh $CLUSTER"
echo ""

print_info "Syncing will download:"
echo "  - checkpoints_${CLUSTER}/"
echo "  - results_${CLUSTER}/"
echo "  - logs_${CLUSTER}/"
echo "  - output_${CLUSTER}/"
echo ""

read -p "Sync results now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Syncing results from $CLUSTER..."
    echo ""

    # You can use --dry-run to see what would be synced first
    read -p "Run in dry-run mode first? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Running dry-run (no files will be copied)..."
        echo ""
        bash "$SCRIPT_DIR/sync_results.sh" "$CLUSTER" --dry-run || print_warning "Dry-run completed with warnings"
        echo ""

        read -p "Proceed with actual sync? (y/n) " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            bash "$SCRIPT_DIR/sync_results.sh" "$CLUSTER"
        fi
    else
        bash "$SCRIPT_DIR/sync_results.sh" "$CLUSTER"
    fi
fi

# ============================================================================
# STEP 6: Summary
# ============================================================================

print_section "Workflow Complete!"

echo "Summary of what you can do next:"
echo ""
echo "1. Review results:"
echo "   ls -la results_${CLUSTER}/"
echo "   cat logs_${CLUSTER}/*.out"
echo ""
echo "2. Submit more jobs:"
echo "   ./submit_job.sh tensor01 another_job train.py --epochs 200"
echo ""
echo "3. Check cluster resources:"
echo "   sshpass -p \"PASSWORD\" ssh user@tensor01.dartmouth.edu nvidia-smi"
echo ""
echo "4. Set up the other cluster:"
echo "   ./setup_cluster.sh tensor02"
echo ""

print_success "Example workflow complete!"
echo ""
