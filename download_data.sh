#!/bin/bash
# Download script for Sherlock fMRI data from Discovery server
#
# Usage: ./download_data.sh
#
# Prerequisites:
# - sshpass installed (brew install sshpass on macOS)
# - Access to discovery.dartmouth.edu
# - Credentials file in ~/.sherlock_credentials

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Sherlock fMRI Data Download Script${NC}"
echo "===================================="
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo -e "${RED}Error: sshpass is not installed${NC}"
    echo "Install it with: brew install sshpass"
    exit 1
fi

# Check if credentials file exists
CRED_FILE="$HOME/.sherlock_credentials"
if [ ! -f "$CRED_FILE" ]; then
    echo -e "${YELLOW}Credentials file not found${NC}"
    echo "Creating $CRED_FILE"
    echo "Please enter your Discovery credentials:"
    read -p "Username: " username
    read -sp "Password: " password
    echo ""
    echo "USERNAME=$username" > "$CRED_FILE"
    echo "PASSWORD=$password" >> "$CRED_FILE"
    chmod 600 "$CRED_FILE"
    echo -e "${GREEN}Credentials saved to $CRED_FILE${NC}"
fi

# Load credentials
source "$CRED_FILE"

# Create data directory
mkdir -p data/sherlock_nii

echo ""
echo -e "${GREEN}Downloading fMRI data from Discovery...${NC}"
echo "Source: discovery.dartmouth.edu:/dartfs/rc/lab/D/DBIC/CDL/data/fMRI/sherlock/nii/"
echo "Destination: data/sherlock_nii/"
echo ""

# Download .nii files
sshpass -p "$PASSWORD" rsync -avz --progress \
    "$USERNAME@discovery.dartmouth.edu:/dartfs/rc/lab/D/DBIC/CDL/data/fMRI/sherlock/nii/*.nii" \
    data/sherlock_nii/

echo ""
echo -e "${GREEN}Download complete!${NC}"
echo ""

# Compress .nii files to .nii.gz
echo -e "${YELLOW}Compressing .nii files to .nii.gz format...${NC}"
for file in data/sherlock_nii/*.nii; do
    if [ -f "$file" ]; then
        echo "  Compressing: $(basename "$file")"
        gzip -9 "$file"
    fi
done

echo ""
echo -e "${GREEN}All files compressed!${NC}"
echo ""
echo "Files in data/sherlock_nii/:"
ls -lh data/sherlock_nii/*.nii.gz | wc -l | xargs echo "Total files:"
du -sh data/sherlock_nii/ | awk '{print "Total size: " $1}'
