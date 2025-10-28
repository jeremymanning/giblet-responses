#!/bin/bash
# Package Sherlock dataset for Dropbox distribution
# Creates a single .zip file containing:
# - Stimulus video (stimuli_Sherlock.m4v)
# - Annotations (annotations.xlsx)
# - All compressed fMRI data (.nii.gz files)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Packaging Sherlock Dataset for Dropbox${NC}"
echo "========================================"
echo ""

# Step 1: Compress .nii files to .nii.gz
echo -e "${YELLOW}Step 1: Compressing .nii files to .nii.gz${NC}"
NII_COUNT=0
for file in data/sherlock_nii/*.nii; do
    if [ -f "$file" ]; then
        NII_COUNT=$((NII_COUNT + 1))
        echo "  Compressing: $(basename "$file")"
        gzip -9 "$file"
    fi
done

if [ $NII_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No .nii files found to compress (maybe already compressed?)${NC}"
fi

# Count .nii.gz files
GZ_COUNT=$(ls -1 data/sherlock_nii/*.nii.gz 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Total .nii.gz files: $GZ_COUNT"
echo ""

# Step 2: Create the package
echo -e "${YELLOW}Step 2: Creating sherlock_dataset.zip${NC}"
echo ""

# Remove old zip if exists
rm -f sherlock_dataset.zip

# Create zip with all data
zip -r sherlock_dataset.zip \
    data/stimuli_Sherlock.m4v \
    data/annotations.xlsx \
    data/sherlock_nii/*.nii.gz

echo ""
echo -e "${GREEN}Package created successfully!${NC}"
echo ""
echo "File: sherlock_dataset.zip"
ls -lh sherlock_dataset.zip
echo ""
echo "Contents:"
unzip -l sherlock_dataset.zip | tail -n 5
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Upload sherlock_dataset.zip to your Dropbox"
echo "2. Get a shareable link (make sure to change dl=0 to dl=1 in the URL)"
echo "3. Update download_data.sh with the new Dropbox link"
echo ""
