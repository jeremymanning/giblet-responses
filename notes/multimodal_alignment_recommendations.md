# Technical Recommendations for Multimodal Neuroscience Autoencoder Project

**Date**: 2025-10-28

## 1. Temporal Alignment of Multimodal Data

### Problem
Aligning data with vastly different sampling rates:
- Video: 25 fps (40 ms intervals)
- Audio: 44.1 kHz (0.023 ms intervals)
- fMRI: 1 Hz (TR=1s)
- Text annotations: irregular intervals

### Standard Approaches from Recent Literature (2024-2025)

#### A. Frame Aggregation Strategy (AFIRE Framework, 2024)
**Recommendation**: Aggregate high-frequency data into TR bins

**Implementation**:
```python
import numpy as np
from scipy.interpolate import interp1d

def align_to_fmri_tr(high_freq_data, high_freq_times, tr_times):
    """
    Aggregate high-frequency data into fMRI TR bins.

    Parameters:
    - high_freq_data: array of shape (n_samples, n_features)
    - high_freq_times: timestamps for high_freq_data
    - tr_times: fMRI TR timestamps

    Returns:
    - tr_aligned_data: array of shape (n_trs, n_features)
    """
    tr_aligned = []
    tr_duration = tr_times[1] - tr_times[0]  # Assuming constant TR

    for tr_time in tr_times:
        # Find all samples within this TR window
        mask = (high_freq_times >= tr_time) & (high_freq_times < tr_time + tr_duration)
        if np.any(mask):
            # Average all samples within TR window
            tr_aligned.append(np.mean(high_freq_data[mask], axis=0))
        else:
            # Interpolate if no samples in window
            tr_aligned.append(np.zeros(high_freq_data.shape[1]))

    return np.array(tr_aligned)
```

**For your project**:
- Video (25 fps → 1 Hz): Aggregate ~25 frames per TR
- Audio features (after mel spectrogram) → aggregate windows into TR bins
- Text annotations: Use nearest-neighbor or forward-fill to TR timestamps

#### B. Hemodynamic Response Function (HRF) Convolution

**Critical**: Account for ~6s hemodynamic lag between stimulus and BOLD response

**Implementation using nilearn**:
```python
from nilearn.glm.first_level import compute_regressor
import numpy as np

def convolve_stimulus_with_hrf(stimulus_times, stimulus_values, tr, n_scans,
                                hrf_model='glover', oversampling=50):
    """
    Convolve stimulus with HRF to predict fMRI response.

    Parameters:
    - stimulus_times: onset times of stimulus events
    - stimulus_values: amplitude/value at each stimulus time
    - tr: repetition time (e.g., 1.0 for 1Hz)
    - n_scans: number of fMRI volumes
    - hrf_model: 'glover', 'spm', or custom function
    - oversampling: temporal oversampling for convolution

    Returns:
    - predicted_bold: convolved signal at TR resolution
    """
    frame_times = np.arange(n_scans) * tr

    # Create condition array: (onset, duration, amplitude)
    conditions = np.column_stack([
        stimulus_times,
        np.ones_like(stimulus_times),  # duration (can vary)
        stimulus_values
    ])

    signal, _ = compute_regressor(
        conditions.T,
        hrf_model,
        frame_times,
        con_id='stimulus',
        oversampling=oversampling
    )

    return signal

# For continuous video/audio stimuli
def continuous_stimulus_to_fmri(features, feature_fps, tr, n_scans):
    """
    Convert continuous stimulus features to HRF-convolved fMRI predictors.

    Parameters:
    - features: array of shape (n_timepoints, n_features)
    - feature_fps: frames per second of features (e.g., 25 for video)
    - tr: fMRI repetition time
    - n_scans: number of fMRI volumes

    Returns:
    - fmri_predictors: array of shape (n_scans, n_features)
    """
    from nilearn.glm.first_level import glover_hrf

    # Create HRF at high temporal resolution
    dt = 1.0 / feature_fps  # sampling period
    hrf_length = 32.0  # seconds
    hrf = glover_hrf(tr=dt, oversampling=1, time_length=hrf_length)

    # Convolve each feature with HRF
    fmri_predictors = []
    for i in range(features.shape[1]):
        convolved = np.convolve(features[:, i], hrf, mode='full')[:len(features)]
        # Downsample to TR
        tr_indices = np.arange(0, len(convolved), int(tr * feature_fps))[:n_scans]
        fmri_predictors.append(convolved[tr_indices])

    return np.column_stack(fmri_predictors)
```

#### C. Dynamic Temporal Alignment (2025 Research)

**For decoding** (fMRI → video), use exponentially weighted multi-frame fusion:

```python
def dynamic_temporal_alignment(fmri_features, video_fps=25, fmri_hz=1,
                                decay_factor=0.8):
    """
    Expand fMRI features to match video frame rate with exponential decay.

    Parameters:
    - fmri_features: array of shape (n_trs, n_features)
    - video_fps: target video frame rate
    - fmri_hz: fMRI sampling rate
    - decay_factor: exponential decay for temporal smoothing

    Returns:
    - video_features: array of shape (n_trs * expansion_factor, n_features)
    """
    expansion_factor = int(video_fps / fmri_hz)
    n_trs, n_features = fmri_features.shape

    video_features = []
    for i in range(n_trs):
        # For each TR, generate multiple video frames
        for frame_offset in range(expansion_factor):
            # Exponentially weighted combination of current and future TRs
            weights = np.array([decay_factor ** j for j in range(min(3, n_trs - i))])
            weights = weights / weights.sum()

            weighted_feature = np.zeros(n_features)
            for j, w in enumerate(weights):
                if i + j < n_trs:
                    weighted_feature += w * fmri_features[i + j]

            video_features.append(weighted_feature)

    return np.array(video_features)
```

### Recommended Library Functions

1. **nilearn.glm.first_level.compute_regressor**: HRF convolution
2. **nilearn.glm.first_level.glover_hrf**: Generate HRF
3. **scipy.interpolate.interp1d**: Linear interpolation for alignment
4. **numpy.interp**: Modern alternative to interp1d for 1D interpolation

### Key Parameters
- **TR**: 1.0 second (your fMRI sampling rate)
- **HRF oversampling**: 50 (for accurate convolution)
- **HRF model**: 'glover' (most common) or 'spm'
- **Hemodynamic lag**: ~6 seconds (handled automatically by HRF)

---

## 2. Audio Processing: Mel Spectrogram Best Practices

### A. Mel Spectrogram Extraction

**Standard configuration for speech/audio reconstruction**:

```python
import librosa
import numpy as np

# Recommended parameters for HiFi-GAN compatibility
MEL_PARAMS = {
    'sr': 22050,           # Sample rate (use 22050 or 44100)
    'n_fft': 1024,         # FFT window size
    'hop_length': 256,      # Hop length (overlap)
    'win_length': 1024,     # Window length
    'n_mels': 80,          # Number of mel bands (80 for HiFi-GAN, 128 for reconstruction)
    'fmin': 0.0,           # Minimum frequency
    'fmax': 8000.0,        # Maximum frequency (8000 for 22050 sr, None for full range)
    'power': 2.0,          # Power spectrogram (magnitude squared)
}

def extract_mel_spectrogram(audio_path, **mel_params):
    """
    Extract mel spectrogram from audio file.

    Returns:
    - mel_spec: mel spectrogram (n_mels, n_frames)
    - mel_spec_db: mel spectrogram in dB scale
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=mel_params['sr'])

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=mel_params['sr'],
        n_fft=mel_params['n_fft'],
        hop_length=mel_params['hop_length'],
        win_length=mel_params['win_length'],
        n_mels=mel_params['n_mels'],
        fmin=mel_params['fmin'],
        fmax=mel_params['fmax'],
        power=mel_params['power']
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec, mel_spec_db

# Usage
mel_spec, mel_spec_db = extract_mel_spectrogram('audio.wav', **MEL_PARAMS)
```

### B. Mel Spectrogram Reconstruction

**Critical**: Direct inversion from mel spectrogram is lossy and produces poor quality

**Three Approaches** (in order of quality):

#### 1. HiFi-GAN Vocoder (RECOMMENDED)

**Best quality, fastest inference**

```python
import torch
import torchaudio
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH

# Load pretrained HiFi-GAN
bundle = HIFIGAN_VOCODER_V3_LJSPEECH
vocoder = bundle.get_vocoder()
vocoder.eval()

def reconstruct_audio_hifigan(mel_spec_db, sample_rate=22050):
    """
    Reconstruct audio from mel spectrogram using HiFi-GAN.

    Parameters:
    - mel_spec_db: mel spectrogram in dB scale (n_mels, n_frames)
    - sample_rate: target sample rate

    Returns:
    - waveform: reconstructed audio (1, n_samples)
    """
    # Convert dB back to power
    mel_spec = librosa.db_to_power(mel_spec_db)

    # Convert to torch tensor
    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, n_frames)

    # Generate audio
    with torch.no_grad():
        waveform = vocoder(mel_tensor)

    return waveform

# Alternative: Use pretrained models from GitHub
# https://github.com/jik876/hifi-gan
```

**For training custom HiFi-GAN**: Use configuration from their repository
- Universal vocoder works across speakers/datasets
- Training requires paired mel-spectrogram + audio data
- Can fine-tune on your specific audio domain

#### 2. WaveGlow (Alternative)

```python
import torch

# Load pretrained WaveGlow from PyTorch Hub
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                           'nvidia_waveglow',
                           model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()

def reconstruct_audio_waveglow(mel_spec):
    """
    Reconstruct audio using WaveGlow.

    Parameters:
    - mel_spec: mel spectrogram (1, n_mels, n_frames)

    Returns:
    - audio: reconstructed waveform
    """
    with torch.no_grad():
        audio = waveglow.infer(mel_spec)

    return audio
```

**Note**: WaveGlow is slower than HiFi-GAN but also produces high-quality audio

#### 3. Griffin-Lim (FALLBACK ONLY)

**Only use if neural vocoders are not feasible** (produces artifacts)

```python
def reconstruct_audio_griffinlim(mel_spec_db, sr=22050, n_fft=1024,
                                  hop_length=256, n_iter=100):
    """
    Reconstruct audio using Griffin-Lim algorithm (low quality).

    Parameters:
    - mel_spec_db: mel spectrogram in dB
    - n_iter: number of Griffin-Lim iterations (more = better quality)

    Returns:
    - audio: reconstructed waveform (1D array)
    """
    # Convert dB to power
    mel_spec = librosa.db_to_power(mel_spec_db)

    # Invert mel spectrogram to linear spectrogram (lossy!)
    spec = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=sr,
        n_fft=n_fft,
        fmin=0.0,
        fmax=8000.0
    )

    # Reconstruct audio using Griffin-Lim
    audio = librosa.griffinlim(
        spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft
    )

    return audio

# Can also use librosa's direct method:
audio = librosa.feature.inverse.mel_to_audio(
    mel_spec,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_iter=100
)
```

### C. Recommended n_mels Parameter

- **For reconstruction with neural vocoders**: 80 mels (HiFi-GAN standard)
- **For higher fidelity**: 128 mels (captures more detail but slower)
- **For fMRI encoding models**: Can use lower (40-80) for dimensionality reduction

### D. Aligning Audio to Video

```python
def align_audio_to_video(audio_path, video_fps=25, mel_params=MEL_PARAMS):
    """
    Extract mel spectrogram aligned to video frame rate.

    Parameters:
    - audio_path: path to audio file
    - video_fps: video frames per second

    Returns:
    - mel_frames: mel spectrogram aligned to video (n_video_frames, n_mels)
    """
    # Extract mel spectrogram
    mel_spec, _ = extract_mel_spectrogram(audio_path, **mel_params)

    # Calculate how many mel frames per video frame
    audio_fps = mel_params['sr'] / mel_params['hop_length']  # ~86 fps for standard params
    frames_per_video = int(audio_fps / video_fps)

    # Aggregate mel frames to match video rate
    n_video_frames = mel_spec.shape[1] // frames_per_video
    mel_frames = []

    for i in range(n_video_frames):
        start_idx = i * frames_per_video
        end_idx = start_idx + frames_per_video
        # Average mel features across video frame duration
        mel_frames.append(np.mean(mel_spec[:, start_idx:end_idx], axis=1))

    return np.array(mel_frames)
```

### Key Recommendations

1. **For reconstruction quality**: Use HiFi-GAN vocoder (not Griffin-Lim)
2. **Standard n_mels**: 80 (for neural vocoders), 128 (for more detail)
3. **Sample rate**: 22050 Hz (standard) or keep original 44100 Hz
4. **Pretrained models**: Use torchaudio.prototype.pipelines for HiFi-GAN
5. **Training vocoders**: Only if you need domain-specific audio (otherwise use pretrained)

---

## 3. Text Embedding and Recovery

### A. Best MTEB Models for Semantic Similarity + Reconstruction

**Top models as of 2024-2025 MTEB Leaderboard**:

1. **sentence-transformers/all-mpnet-base-v2** (Strong overall performance)
   - 768-dimensional embeddings
   - Good semantic similarity
   - Wide community support

2. **BAAI/bge-large-en-v1.5** (State-of-the-art)
   - 1024-dimensional embeddings
   - Top performance on MTEB
   - Excellent for retrieval

3. **intfloat/e5-large-v2** (Excellent for similarity)
   - 1024-dimensional embeddings
   - Strong semantic understanding

4. **NV-Embed-v2** (NVIDIA, multimodal-aware)
   - Based on Mistral 7B
   - Generalist embedding model
   - Excellent for cross-modal tasks

**Implementation**:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Or: model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def embed_text_annotations(texts):
    """
    Embed text annotations using sentence transformers.

    Parameters:
    - texts: list of strings

    Returns:
    - embeddings: array of shape (n_texts, embedding_dim)
    """
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# For semantic similarity
from sentence_transformers.util import cos_sim

def find_similar_texts(query_text, candidate_texts, top_k=5):
    """Find most similar texts to query."""
    query_emb = model.encode(query_text, convert_to_numpy=True)
    candidate_embs = model.encode(candidate_texts, convert_to_numpy=True)

    similarities = cos_sim(query_emb, candidate_embs)
    top_indices = similarities.argsort(descending=True)[:top_k]

    return [candidate_texts[i] for i in top_indices]
```

### B. Embedding to Text Recovery

**Critical**: Direct embedding-to-text is extremely challenging

**Three Approaches**:

#### 1. Vec2Text Library (RECOMMENDED)

**State-of-the-art for embedding inversion** (2024)

```python
# Install: pip install vec2text
import vec2text

# Load inversion model (requires training or pretrained weights)
corrector = vec2text.load_corrector("text-embedding-ada-002")  # or other models

def invert_embeddings_to_text(embeddings, model_name="text-embedding-ada-002"):
    """
    Invert embeddings back to text using vec2text.

    Parameters:
    - embeddings: array of shape (n_texts, embedding_dim)
    - model_name: name of embedding model used

    Returns:
    - reconstructed_texts: list of reconstructed strings
    """
    # Load appropriate corrector
    corrector = vec2text.load_corrector(model_name)

    # Invert embeddings
    reconstructed = vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=corrector,
        num_steps=20  # iterative refinement steps
    )

    return reconstructed

# GitHub: https://github.com/vec2text/vec2text
```

**Key features**:
- Supports sentence-transformers and OpenAI embeddings
- Iterative refinement for better reconstruction
- Can recover semantic meaning (not exact text)

#### 2. Nearest Neighbor Retrieval (PRACTICAL ALTERNATIVE)

**For irregular text annotations in your project**:

```python
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class TextEmbeddingDatabase:
    """Store and retrieve text from embeddings."""

    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.embeddings = None
        self.nn_index = None

    def add_texts(self, texts):
        """Add texts to database."""
        self.texts.extend(texts)
        new_embeddings = self.model.encode(texts)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Build nearest neighbor index
        self.nn_index = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn_index.fit(self.embeddings)

    def retrieve_from_embedding(self, embedding, top_k=5):
        """Retrieve most similar text given an embedding."""
        distances, indices = self.nn_index.kneighbors(
            embedding.reshape(1, -1),
            n_neighbors=top_k
        )

        return [self.texts[i] for i in indices[0]]

    def decode_embeddings(self, embeddings):
        """Decode multiple embeddings to texts."""
        texts = []
        for emb in embeddings:
            retrieved = self.retrieve_from_embedding(emb, top_k=1)
            texts.append(retrieved[0])
        return texts

# Usage for your project
text_db = TextEmbeddingDatabase()

# Add all text annotations from your dataset
all_annotations = ["Person walks into room", "Door closes", "Music starts", ...]
text_db.add_texts(all_annotations)

# Later: decode from predicted embeddings
predicted_text = text_db.retrieve_from_embedding(predicted_embedding)
```

#### 3. Generative Decoder (RESEARCH/ADVANCED)

**Train a decoder model** (for semantic reconstruction, not exact recovery):

```python
import torch
import torch.nn as nn

class EmbeddingToTextDecoder(nn.Module):
    """
    Decoder to generate text from embeddings.
    Requires training with a language model head.
    """
    def __init__(self, embedding_dim=768, vocab_size=30522, hidden_dim=1024):
        super().__init__()

        # Project embedding to sequence
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, embeddings, max_length=50):
        """
        Generate text from embeddings.

        Parameters:
        - embeddings: (batch, embedding_dim)
        - max_length: maximum sequence length

        Returns:
        - token_ids: (batch, max_length)
        """
        # Project embeddings
        memory = self.embedding_proj(embeddings).unsqueeze(1)  # (batch, 1, hidden)

        # Autoregressive generation (simplified)
        # Full implementation would use beam search
        batch_size = embeddings.size(0)
        generated = torch.zeros(batch_size, max_length, dtype=torch.long)

        # ... (autoregressive decoding loop)

        return generated

# Note: Training this requires large paired datasets of text + embeddings
# Consider using pretrained language model decoders (GPT-2, T5, etc.)
```

### C. Practical Recommendations for Your Project

**For irregular text annotations**:

1. **Encoding phase**:
   - Use `sentence-transformers/all-mpnet-base-v2` or `BAAI/bge-large-en-v1.5`
   - Embed all annotations at their original timestamps
   - Align to fMRI TR using nearest-neighbor or forward-fill

2. **Decoding phase** (embedding → text):
   - **Option A**: Use nearest-neighbor retrieval with original annotation database
   - **Option B**: Use vec2text for semantic reconstruction
   - **Option C**: Generate word clouds from embeddings (for visualization only)

3. **Word cloud generation** (visualization):
```python
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def embedding_to_wordcloud(embedding, text_corpus, model, top_n=50):
    """
    Generate word cloud from embedding using similarity to corpus.

    Parameters:
    - embedding: single embedding vector
    - text_corpus: list of reference texts
    - model: sentence transformer model
    - top_n: number of top similar texts to use

    Returns:
    - wordcloud: WordCloud object
    """
    # Get corpus embeddings
    corpus_embeddings = model.encode(text_corpus)

    # Find most similar texts
    similarities = cosine_similarity([embedding], corpus_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Combine top texts
    combined_text = ' '.join([text_corpus[i] for i in top_indices])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400).generate(combined_text)

    return wordcloud
```

### Key Recommendations

1. **Best model**: `BAAI/bge-large-en-v1.5` (state-of-the-art) or `all-mpnet-base-v2` (balanced)
2. **Reconstruction**: Use nearest-neighbor retrieval with original text database
3. **For semantic recovery**: Use vec2text library (https://github.com/vec2text/vec2text)
4. **Visualization**: Generate word clouds from similar texts in corpus
5. **No exact recovery**: Embedding → text is semantically lossy (embeddings preserve meaning, not exact wording)

---

## 4. fMRI Data Formatting

### A. CMU Format (BrainIAK HTFA)

**Definition**: "CMU format" refers to the data shape expected by BrainIAK's HTFA

**Key difference from nilearn**:
- **nilearn format**: (n_timepoints, n_voxels) - rows are time
- **BrainIAK/CMU format**: (n_voxels, n_timepoints) - rows are voxels

**Conversion example**:

```python
import numpy as np
import nibabel as nib
from nilearn import masking

def nifti_to_cmu_format(nifti_file, mask_img=None):
    """
    Convert NIfTI file to CMU format for BrainIAK.

    Parameters:
    - nifti_file: path to 4D NIfTI file
    - mask_img: brain mask (optional, will compute if None)

    Returns:
    - data_cmu: array of shape (n_voxels, n_timepoints)
    - coords: voxel coordinates (n_voxels, 3)
    """
    # Load data
    img = nib.load(nifti_file)

    # Apply masking
    if mask_img is None:
        from nilearn.masking import compute_brain_mask
        mask_img = compute_brain_mask(img)

    # Extract masked data (nilearn format: n_timepoints x n_voxels)
    data_nilearn = masking.apply_mask(img, mask_img)

    # Transpose to CMU format (n_voxels x n_timepoints)
    data_cmu = data_nilearn.T

    # Get voxel coordinates
    mask_data = mask_img.get_fdata()
    coords = np.column_stack(np.where(mask_data))

    return data_cmu, coords


def cmu_to_nilearn_format(data_cmu):
    """Convert CMU format to nilearn format."""
    return data_cmu.T


# For multiple subjects
def prepare_htfa_data(nifti_files, shared_mask=None):
    """
    Prepare data for BrainIAK HTFA.

    Parameters:
    - nifti_files: list of paths to subject NIfTI files
    - shared_mask: shared brain mask across subjects

    Returns:
    - htfa_data: list of dicts with 'Z' (data) and 'R' (coords) for each subject
    """
    htfa_data = []

    for nifti_file in nifti_files:
        data_cmu, coords = nifti_to_cmu_format(nifti_file, shared_mask)

        htfa_data.append({
            'Z': data_cmu,      # (n_voxels, n_timepoints)
            'R': coords         # (n_voxels, 3) - scanner coordinates
        })

    return htfa_data

# Usage with BrainIAK
from brainiak.factoranalysis.htfa import HTFA

# Prepare data
htfa_data = prepare_htfa_data(subject_nifti_files, shared_mask)

# Fit HTFA
htfa = HTFA(n_iter=100, n_nodes=10)
htfa.fit(
    [x['Z'] for x in htfa_data],  # List of (voxels, timepoints) arrays
    [x['R'] for x in htfa_data]   # List of coordinate arrays
)
```

### B. Creating Shared Brain Masks Across Subjects

**Recommended approach using nilearn**:

```python
from nilearn.masking import compute_multi_brain_mask, compute_multi_epi_mask
from nilearn.image import resample_to_img
import nibabel as nib

def create_shared_brain_mask(subject_files, mask_type='epi', threshold=0.5):
    """
    Create shared brain mask across multiple subjects.

    Parameters:
    - subject_files: list of paths to subject NIfTI files
    - mask_type: 'epi' for functional data, 'brain' for anatomical
    - threshold: fraction of subjects that must have brain at each voxel

    Returns:
    - shared_mask: nibabel image with shared brain mask
    """
    if mask_type == 'epi':
        # For functional (EPI) images
        shared_mask = compute_multi_epi_mask(
            subject_files,
            threshold=threshold,
            opening=2,  # morphological opening (removes small holes)
            connected=True  # keep only largest connected component
        )
    else:
        # For anatomical or general brain masks
        shared_mask = compute_multi_brain_mask(
            subject_files,
            threshold=threshold,
            mask_type='whole-brain'  # or 'gm' for gray matter
        )

    return shared_mask


# Alternative: Use MNI template mask
from nilearn.datasets import load_mni152_brain_mask, load_mni152_gm_mask
from nilearn.image import resample_img

def create_mni_template_mask(reference_img, mask_type='whole-brain'):
    """
    Create mask based on MNI152 template, resampled to subject space.

    Parameters:
    - reference_img: reference image (e.g., one subject's functional data)
    - mask_type: 'whole-brain' or 'gm' (gray matter)

    Returns:
    - mask_img: nibabel image with mask in subject space
    """
    # Load MNI template mask
    if mask_type == 'gm':
        mni_mask = load_mni152_gm_mask()
    else:
        mni_mask = load_mni152_brain_mask()

    # Resample to reference image space
    mask_img = resample_img(
        mni_mask,
        target_affine=reference_img.affine,
        target_shape=reference_img.shape[:3],
        interpolation='nearest'
    )

    return mask_img


# Recommended workflow for multi-subject analysis
def prepare_multisubject_data(subject_files, use_mni=False):
    """
    Prepare aligned data for multi-subject analysis.

    Parameters:
    - subject_files: list of NIfTI file paths
    - use_mni: if True, align all subjects to MNI space first

    Returns:
    - masked_data: list of (n_voxels, n_timepoints) arrays
    - shared_mask: nibabel image with shared mask
    """
    if use_mni:
        # Option 1: Register all subjects to MNI template
        from nilearn.image import resample_to_img

        # Load MNI template
        from nilearn.datasets import load_mni152_template
        mni_template = load_mni152_template()

        # Resample all subjects to MNI (assumes preprocessing already done)
        # This step typically done with fMRIPrep or FSL FLIRT
        reference_img = nib.load(subject_files[0])
        shared_mask = create_mni_template_mask(reference_img)
    else:
        # Option 2: Create shared mask in native space
        shared_mask = create_shared_brain_mask(subject_files, threshold=0.5)

    # Extract masked data for each subject
    masked_data = []
    for subj_file in subject_files:
        data_cmu, _ = nifti_to_cmu_format(subj_file, shared_mask)
        masked_data.append(data_cmu)

    return masked_data, shared_mask
```

### C. Handling Inter-Subject Variability

**Three standard approaches**:

#### 1. Shared Response Model (SRM) - BrainIAK

**Maps individual brains to shared space**:

```python
from brainiak.funcalign.srm import SRM

def apply_srm_alignment(subject_data_list, n_features=50):
    """
    Apply Shared Response Model for inter-subject alignment.

    Parameters:
    - subject_data_list: list of arrays (n_voxels, n_timepoints) per subject
    - n_features: dimensionality of shared space

    Returns:
    - srm_model: trained SRM model
    - shared_data: list of transformed data in shared space
    """
    # Initialize SRM
    srm = SRM(n_iter=10, features=n_features)

    # Fit SRM (learns transformation for each subject)
    srm.fit(subject_data_list)

    # Transform data to shared space
    shared_data = srm.transform(subject_data_list)

    return srm, shared_data


# For new subjects
def transform_new_subject(srm_model, new_subject_data):
    """Transform new subject data to shared space."""
    return srm_model.transform([new_subject_data])[0]
```

#### 2. Hyperalignment

**More flexible than SRM, can align subsets of voxels**:

```python
from brainiak.funcalign.rsrm import RSRM

def apply_hyperalignment(subject_data_list, n_iter=10):
    """
    Apply hyperalignment across subjects.

    Parameters:
    - subject_data_list: list of (n_voxels, n_timepoints) arrays
    - n_iter: number of iterations

    Returns:
    - aligned_data: list of aligned data in common space
    """
    # RSRM is regularized SRM with more robustness
    rsrm = RSRM(n_iter=n_iter, features=100)
    rsrm.fit(subject_data_list)

    aligned_data = rsrm.transform(subject_data_list)

    return aligned_data
```

#### 3. Template-Based (MNI Space)

**Most common for group analysis**:

```python
def register_to_mni(subject_file, output_file):
    """
    Register subject to MNI space using nilearn.

    Note: For production, use fMRIPrep or FSL/ANTs for registration
    This is a simplified example.

    Parameters:
    - subject_file: path to subject NIfTI
    - output_file: path to save registered data

    Returns:
    - registered_img: nibabel image in MNI space
    """
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img

    # Load subject and template
    subject_img = nib.load(subject_file)
    mni_template = load_mni152_template()

    # Resample to MNI space
    # Note: This is only resampling, not registration!
    # Real registration requires fMRIPrep, FSL FLIRT/FNIRT, or ANTs
    registered_img = resample_to_img(
        subject_img,
        mni_template,
        interpolation='continuous'
    )

    # Save
    nib.save(registered_img, output_file)

    return registered_img

# Better: Use fMRIPrep outputs which are already in MNI space
# fMRIPrep outputs: sub-01_task-movie_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
```

### D. Complete Example: Multi-Subject Analysis Pipeline

```python
from nilearn.masking import apply_mask, compute_multi_epi_mask
from brainiak.funcalign.srm import SRM
import numpy as np

def complete_multisubject_pipeline(subject_files, n_shared_features=100):
    """
    Complete pipeline for multi-subject fMRI analysis.

    Parameters:
    - subject_files: list of preprocessed NIfTI files
    - n_shared_features: dimensionality of shared space

    Returns:
    - aligned_data: data in shared space (n_subjects, n_features, n_timepoints)
    - srm_model: trained SRM model
    - shared_mask: brain mask
    """
    # Step 1: Create shared brain mask
    print("Creating shared brain mask...")
    shared_mask = compute_multi_epi_mask(
        subject_files,
        threshold=0.8,  # 80% of subjects must have brain signal
        opening=2,
        connected=True
    )

    # Step 2: Extract masked data
    print("Extracting masked data...")
    subject_data = []
    for subj_file in subject_files:
        # Load and mask data
        data_nilearn = apply_mask(subj_file, shared_mask)  # (n_timepoints, n_voxels)
        data_cmu = data_nilearn.T  # Convert to (n_voxels, n_timepoints)
        subject_data.append(data_cmu)

    # Step 3: Apply SRM for inter-subject alignment
    print("Applying Shared Response Model...")
    srm = SRM(n_iter=10, features=n_shared_features)
    srm.fit(subject_data)
    aligned_data = srm.transform(subject_data)

    # aligned_data is list of (n_features, n_timepoints) arrays
    aligned_data = np.array(aligned_data)  # (n_subjects, n_features, n_timepoints)

    print(f"Final aligned data shape: {aligned_data.shape}")
    print(f"Shared mask has {np.sum(shared_mask.get_fdata())} voxels")

    return aligned_data, srm, shared_mask


# Usage
subject_files = [
    '/path/to/sub-01_bold.nii.gz',
    '/path/to/sub-02_bold.nii.gz',
    '/path/to/sub-03_bold.nii.gz',
]

aligned_data, srm_model, mask = complete_multisubject_pipeline(subject_files)
```

### Key Recommendations

1. **CMU format**: Transpose nilearn data (n_timepoints, n_voxels) → (n_voxels, n_timepoints)
2. **Shared mask**: Use `compute_multi_epi_mask()` with threshold=0.5-0.8
3. **Inter-subject alignment**:
   - For functional alignment: Use SRM or RSRM (BrainIAK)
   - For anatomical alignment: Use fMRIPrep → MNI space
   - For within-subject: Standard preprocessing (motion correction, etc.)
4. **Library functions**:
   - `brainiak.funcalign.srm.SRM`: Shared Response Model
   - `nilearn.masking.compute_multi_epi_mask`: Multi-subject masking
   - `nilearn.datasets.load_mni152_brain_mask`: Template masks
5. **Data flow**:
   - fMRIPrep (preprocessing) → nilearn (masking) → BrainIAK (alignment/analysis)

---

## Summary: Recommended Tech Stack

### Core Libraries

1. **fMRI processing**:
   - `nibabel`: I/O for NIfTI files
   - `nilearn`: Masking, preprocessing, GLM
   - `brainiak`: Advanced analysis (SRM, HTFA, ISC)

2. **Audio processing**:
   - `librosa`: Mel spectrogram extraction
   - `torchaudio`: HiFi-GAN vocoder (reconstruction)
   - `torch`: Neural vocoder models

3. **Text embedding**:
   - `sentence-transformers`: MTEB models
   - `vec2text`: Embedding inversion
   - `transformers`: Underlying models

4. **Temporal alignment**:
   - `scipy.interpolate`: Interpolation
   - `numpy`: Array operations
   - `nilearn.glm.first_level`: HRF convolution

### Installation

```bash
# Core fMRI
pip install nibabel nilearn brainiak

# Audio
pip install librosa torchaudio soundfile

# Text
pip install sentence-transformers vec2text transformers

# General
pip install numpy scipy scikit-learn matplotlib
```

### Critical Parameters Summary

| Task | Parameter | Recommended Value |
|------|-----------|------------------|
| fMRI sampling | TR | 1.0 s |
| Video | FPS | 25 |
| Audio sample rate | sr | 22050 Hz |
| Mel bands | n_mels | 80 (HiFi-GAN) / 128 (detail) |
| FFT size | n_fft | 1024 |
| Hop length | hop_length | 256 |
| Text embedding | model | 'BAAI/bge-large-en-v1.5' |
| Embedding dim | - | 768-1024 |
| HRF model | - | 'glover' |
| HRF oversampling | - | 50 |
| Shared mask threshold | - | 0.5-0.8 |
| SRM features | n_features | 50-100 |

---

## References and Further Reading

1. **Temporal Alignment**:
   - AFIRE Framework (2024): Multimodal fMRI encoding
   - Dynamic Temporal Alignment (2025): fMRI-to-video decoding
   - Nilearn documentation: https://nilearn.github.io

2. **Audio Reconstruction**:
   - HiFi-GAN: https://github.com/jik876/hifi-gan
   - WaveGlow: https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/
   - Librosa: https://librosa.org

3. **Text Embeddings**:
   - MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
   - Vec2Text: https://github.com/vec2text/vec2text
   - Sentence Transformers: https://www.sbert.net

4. **BrainIAK**:
   - HTFA Tutorial: https://brainiak.org/examples/htfa.html
   - SRM Documentation: https://brainiak.org/docs/brainiak.funcalign.html
   - BrainIAK Tutorials: https://brainiak.org/tutorials/

---

## Notes

- These recommendations are based on current best practices as of October 2025
- Always validate on your specific dataset
- Consider computational resources when choosing model sizes
- For production, use fMRIPrep for preprocessing
- Test different configurations for your specific use case
