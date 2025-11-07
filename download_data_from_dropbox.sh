#!/bin/bash
# Download Sherlock dataset from Dropbox
# Simple download and extract - no credentials needed

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Sherlock Dataset Download (from Dropbox)${NC}"
echo "=========================================="
echo ""

# Dropbox URL
DROPBOX_URL="https://www.dropbox.com/scl/fi/tlr6orirwc14gdq7yqwcl/sherlock_dataset.zip?rlkey=82h9hyrbv37xvtqff5mw6h6m5&dl=1"

# Download the dataset
echo -e "${YELLOW}Downloading sherlock_dataset.zip from Dropbox...${NC}"
echo ""

curl -L "$DROPBOX_URL" -o sherlock_dataset.zip

# Check if download was successful
if [ ! -f sherlock_dataset.zip ] || [ ! -s sherlock_dataset.zip ]; then
    echo -e "${RED}Download failed or file is empty${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Download complete!${NC}"
ls -lh sherlock_dataset.zip
echo ""

# Extract the dataset
echo -e "${YELLOW}Extracting dataset...${NC}"
echo ""

unzip -o sherlock_dataset.zip

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Dataset contents:"
echo "  - data/stimuli_Sherlock.m4v (Sherlock episode video)"
echo "  - data/annotations.xlsx (Scene annotations)"
echo "  - data/sherlock_nii/*.nii.gz (fMRI data for 17 subjects)"
echo ""

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
rm sherlock_dataset.zip
echo ""
echo -e "${GREEN}All done!${NC}"
