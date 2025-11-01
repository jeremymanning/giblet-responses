#!/usr/bin/env python3
"""Test 12kHz downsampling with EnCodec 3.0 kbps"""

import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import EncodecModel, AutoProcessor
from pesq import pesq
from pystoi import stoi

print("Testing 12kHz + 3.0 kbps EnCodec")
print("=" * 60)

# Load model
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
model.set_target_bandwidth(3.0)

# Load audio
audio, sr = librosa.load("data/stimuli_Sherlock.m4v", sr=48000, duration=10, mono=True)

# Test 1: 24kHz
audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=24000)
inputs_24k = processor(raw_audio=audio_24k, sampling_rate=24000, return_tensors="pt")
with torch.no_grad():
    enc_24k = model.encode(inputs_24k["input_values"], inputs_24k["padding_mask"])
    dec_24k = model.decode(enc_24k[0], enc_24k[1])[0]
recon_24k = dec_24k.cpu().squeeze().numpy()

# Test 2: 12kHz
audio_12k = librosa.resample(audio, orig_sr=sr, target_sr=12000)
audio_12k_up = librosa.resample(audio_12k, orig_sr=12000, target_sr=24000)
inputs_12k = processor(raw_audio=audio_12k_up, sampling_rate=24000, return_tensors="pt")
with torch.no_grad():
    enc_12k = model.encode(inputs_12k["input_values"], inputs_12k["padding_mask"])
    dec_12k = model.decode(enc_12k[0], enc_12k[1])[0]
recon_12k = dec_12k.cpu().squeeze().numpy()

# Metrics
ref = audio_24k[:len(recon_24k)]
pesq_24k = pesq(24000, ref, recon_24k, 'nb')
stoi_24k = stoi(ref, recon_24k, 24000)
snr_24k = 10 * np.log10(np.sum(ref**2) / np.sum((ref - recon_24k)**2))

ref = audio_24k[:len(recon_12k)]
pesq_12k = pesq(24000, ref, recon_12k, 'nb')
stoi_12k = stoi(ref, recon_12k, 24000)
snr_12k = 10 * np.log10(np.sum(ref**2) / np.sum((ref - recon_12k)**2))

print("\n24kHz results:")
print(f"  PESQ: {pesq_24k:.3f}")
print(f"  STOI: {stoi_24k:.3f}")
print(f"  SNR: {snr_24k:.2f} dB")
print(f"  Frames: {enc_24k[0].shape[2]}")

print("\n12kHz results:")
print(f"  PESQ: {pesq_12k:.3f}")
print(f"  STOI: {stoi_12k:.3f}")
print(f"  SNR: {snr_12k:.2f} dB")
print(f"  Frames: {enc_12k[0].shape[2]}")

# Save
Path("encodec_12khz_test").mkdir(exist_ok=True)
sf.write("encodec_12khz_test/reconstructed_24khz_3kbps.wav", recon_24k, 24000)
sf.write("encodec_12khz_test/reconstructed_12khz_3kbps.wav", recon_12k, 24000)
print(f"\nWAV files saved to encodec_12khz_test/")
print("Please listen and compare!")
