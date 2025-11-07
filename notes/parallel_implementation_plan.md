# Parallel Implementation Plan

## Task Dependencies

### Batch 1: Independent modules (CAN RUN IN PARALLEL)
1. **Text processing** (`giblet/data/text.py`)
   - Depends on: annotations.xlsx, sentence-transformers
   - Outputs: Text embeddings (n_trs, 1024)

2. **fMRI processing** (`giblet/data/fmri.py`)
   - Depends on: .nii.gz files, nibabel, nilearn
   - Outputs: Brain timeseries (n_trs, 83300), voxel coordinates

3. **HRF convolution** (`giblet/alignment/hrf.py`)
   - Depends on: nilearn
   - Outputs: HRF convolution function

### Batch 2: Depends on Batch 1
4. **Temporal sync** (`giblet/alignment/sync.py`)
   - Depends on: All data processors
   
5. **Encoder** (`giblet/models/encoder.py`)
   - Depends on: Input dimensions from processors

6. **Decoder** (`giblet/models/decoder.py`)
   - Depends on: Output dimensions from processors

### Batch 3: Depends on Batch 2
7. **Autoencoder** (`giblet/models/autoencoder.py`)
   - Depends on: Encoder + Decoder

8. **Dataset** (`giblet/data/dataset.py`)
   - Depends on: All processors + alignment

### Batch 4: Depends on Batch 3
9. **Training loop** (`giblet/training/trainer.py`)
   - Depends on: Dataset + Autoencoder

10. **Integration tests** (`tests/integration/`)
    - Depends on: Everything

11. **Cluster setup scripts**
    - Independent but uses all modules
