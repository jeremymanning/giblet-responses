# Data Setup Instructions

## For Lab Coordinator (One-Time Setup)

After downloading the fMRI data from Discovery server:

1. **Package the dataset**:
   ```bash
   chmod +x package_for_dropbox.sh
   ./package_for_dropbox.sh
   ```

   This creates `sherlock_dataset.zip` containing:
   - `data/stimuli_Sherlock.m4v` (272 MB)
   - `data/annotations.xlsx` (173 KB)
   - `data/sherlock_nii/*.nii.gz` (17 compressed fMRI files, ~12 GB total)

2. **Upload to Dropbox**:
   - Upload `sherlock_dataset.zip` to your Dropbox
   - Right-click → Share → Create link
   - Copy the link and **change `dl=0` to `dl=1`** at the end
   - Example: `https://www.dropbox.com/scl/fi/xxxxx/sherlock_dataset.zip?rlkey=xxxxx&dl=1`

3. **Update download script**:
   - Edit `download_data_from_dropbox.sh`
   - Replace `PLACEHOLDER_DROPBOX_URL` with your actual Dropbox link
   - Commit the updated script to GitHub

## For Lab Members

Simple one-command setup:

```bash
chmod +x download_data_from_dropbox.sh
./download_data_from_dropbox.sh
```

This will:
1. Download `sherlock_dataset.zip` from Dropbox
2. Extract all data files to the correct locations
3. Clean up the zip file

**Requirements**: Only `curl` and `unzip` (standard on macOS/Linux)

## Alternative: Direct from Discovery Server

If you have access to Discovery:

```bash
chmod +x download_data.sh
./download_data.sh
```

This downloads directly from `/dartfs/rc/lab/D/DBIC/CDL/data/fMRI/sherlock/nii/` on Discovery.

**Requirements**: `sshpass` and Discovery credentials

## Dataset Structure

After setup, you'll have:

```
data/
├── stimuli_Sherlock.m4v              # 272 MB - Sherlock episode video
├── annotations.xlsx                   # 173 KB - Scene-level annotations
└── sherlock_nii/                      # ~12 GB compressed
    ├── sherlock_movie_s1.nii.gz      # Subject 1
    ├── sherlock_movie_s10.nii.gz     # Subject 10
    ├── sherlock_movie_s11.nii.gz     # Subject 11
    ├── ...                           # Subjects 12-16
    ├── sherlock_movie_s2.nii.gz      # Subject 2
    ├── ...                           # Subjects 3-9
    └── sherlock_movie_s9.nii.gz      # Subject 9
```

Total size: ~12-13 GB

## Notes

- Data files are in `.gitignore` and will not be committed to GitHub
- Only the download scripts and documentation are version controlled
- The Dropbox link is embedded in `download_data_from_dropbox.sh` for easy sharing
