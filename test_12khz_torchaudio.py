import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path

print("Testing 12kHz vs 24kHz audio quality")

# Load from existing test outputs (avoid reprocessing)
speech_24k_3kbps = librosa.load("encodec_parameter_sweep/speech_encodec_24khz_bw3.0kbps.wav", sr=24000)[0]
music_24k_3kbps = librosa.load("encodec_parameter_sweep/music_encodec_24khz_bw3.0kbps.wav", sr=24000)[0]

# Downsample to 12kHz
speech_12k = librosa.resample(speech_24k_3kbps, orig_sr=24000, target_sr=12000)
music_12k = librosa.resample(music_24k_3kbps, orig_sr=24000, target_sr=12000)

# Save
Path("encodec_12khz_test").mkdir(exist_ok=True)
sf.write("encodec_12khz_test/speech_24khz_3kbps.wav", speech_24k_3kbps, 24000)
sf.write("encodec_12khz_test/speech_12khz_downsampled.wav", speech_12k, 12000)
sf.write("encodec_12khz_test/music_24khz_3kbps.wav", music_24k_3kbps, 24000)
sf.write("encodec_12khz_test/music_12khz_downsampled.wav", music_12k, 12000)

print("Files saved to encodec_12khz_test/")
print("\nPlease listen and compare:")
print("  speech_24khz_3kbps.wav (current approved)")
print("  speech_12khz_downsampled.wav (more efficient option)")
