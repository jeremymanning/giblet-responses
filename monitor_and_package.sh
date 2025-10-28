#!/bin/bash
# Monitor rsync download and automatically package when complete

echo "Monitoring rsync download..."
echo "Will automatically package dataset when complete."
echo ""

# Wait for the background rsync process to complete
# We're looking for 17 .nii files to be downloaded

TARGET_FILES=17
while true; do
    # Count .nii files
    COUNT=$(ls -1 sherlock_nii/*.nii 2>/dev/null | wc -l | tr -d ' ')

    if [ "$COUNT" -eq "$TARGET_FILES" ]; then
        echo ""
        echo "Download complete! All $TARGET_FILES files downloaded."
        echo ""
        break
    fi

    echo -ne "\rFiles downloaded: $COUNT/$TARGET_FILES"
    sleep 10
done

# Give rsync a moment to finish writing the last file
sleep 5

# Run the packaging script
echo "Starting packaging process..."
echo ""
chmod +x package_for_dropbox.sh
./package_for_dropbox.sh

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload sherlock_dataset.zip to your Dropbox"
echo "2. Get shareable link and change dl=0 to dl=1"
echo "3. Update download_data_from_dropbox.sh with the link"
echo "4. Commit scripts and documentation to GitHub"
