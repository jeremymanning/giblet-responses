#!/bin/bash
# Compress .nii files to .nii.gz format
# This saves significant disk space (~50-70% compression typical for neuroimaging data)

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Compressing NIfTI files to .nii.gz format${NC}"
echo "==========================================="
echo ""

# Find all .nii files
NII_FILES=(sherlock_nii/*.nii)

if [ ! -e "${NII_FILES[0]}" ]; then
    echo "No .nii files found in sherlock_nii/"
    exit 0
fi

TOTAL=${#NII_FILES[@]}
CURRENT=0

for file in "${NII_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo -e "${YELLOW}[$CURRENT/$TOTAL]${NC} Compressing: $(basename "$file")"

    # Get file size before compression
    SIZE_BEFORE=$(du -h "$file" | cut -f1)

    # Compress with gzip (maximum compression)
    gzip -9 "$file"

    # Get compressed size
    SIZE_AFTER=$(du -h "${file}.gz" | cut -f1)

    echo "  Before: $SIZE_BEFORE  →  After: $SIZE_AFTER"
done

echo ""
echo -e "${GREEN}Compression complete!${NC}"
echo ""
echo "Summary:"
ls -lh sherlock_nii/*.nii.gz | awk '{print $9, $5}'
