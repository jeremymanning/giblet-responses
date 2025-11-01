# EnCodec Parameter Comparison

## Summary Statistics

### By Bandwidth (24kHz model, averaged across segments)

|   bandwidth_kbps |   snr_db |   pesq |   stoi |   compression_ratio |   total_time_sec |
|-----------------:|---------:|-------:|-------:|--------------------:|-----------------:|
|              1.5 |    1.659 |  1.497 |  0.649 |                 320 |            0.478 |
|              3   |    3.474 |  1.942 |  0.737 |                 160 |            0.339 |
|              6   |    5.937 |  2.664 |  0.825 |                  80 |            0.35  |
|             12   |    8.155 |  3.557 |  0.895 |                  40 |            0.359 |
|             24   |    9.344 |  4.033 |  0.937 |                  20 |            0.373 |

## Detailed Results

| segment   | model         |   bandwidth_kbps |   snr_db |   pesq |   stoi |   compression_ratio |   total_time_sec |
|:----------|:--------------|-----------------:|---------:|-------:|-------:|--------------------:|-----------------:|
| speech    | encodec_24khz |              1.5 |     0.59 |  1.427 |  0.73  |                 320 |            0.74  |
| speech    | encodec_24khz |              3   |     2.44 |  1.909 |  0.804 |                 160 |            0.334 |
| speech    | encodec_24khz |              6   |     4.93 |  2.652 |  0.879 |                  80 |            0.339 |
| speech    | encodec_24khz |             12   |     7.27 |  3.431 |  0.932 |                  40 |            0.367 |
| speech    | encodec_24khz |             24   |     8.51 |  3.867 |  0.959 |                  20 |            0.362 |
| speech    | encodec_48khz |              3   |     3.35 |  1.473 |  0.772 |                1600 |            0.916 |
| speech    | encodec_48khz |              6   |     5.8  |  2.329 |  0.85  |                 800 |            0.92  |
| speech    | encodec_48khz |             12   |     8.81 |  3.018 |  0.91  |                 400 |            0.931 |
| speech    | encodec_48khz |             24   |    12.01 |  3.599 |  0.949 |                 200 |            0.964 |
| music     | encodec_24khz |              1.5 |     0.42 |  1.424 |  0.573 |                 320 |            0.323 |
| music     | encodec_24khz |              3   |     1.91 |  1.789 |  0.662 |                 160 |            0.339 |
| music     | encodec_24khz |              6   |     3.96 |  2.401 |  0.759 |                  80 |            0.363 |
| music     | encodec_24khz |             12   |     5.72 |  3.314 |  0.844 |                  40 |            0.363 |
| music     | encodec_24khz |             24   |     6.63 |  3.966 |  0.904 |                  20 |            0.379 |
| music     | encodec_48khz |              3   |     3.2  |  1.839 |  0.666 |                1600 |            0.903 |
| music     | encodec_48khz |              6   |     5.16 |  2.498 |  0.775 |                 800 |            0.927 |
| music     | encodec_48khz |             12   |     7.32 |  3.268 |  0.866 |                 400 |            0.941 |
| music     | encodec_48khz |             24   |     9.33 |  3.858 |  0.93  |                 200 |            0.974 |
| mixed     | encodec_24khz |              1.5 |     3.96 |  1.641 |  0.645 |                 320 |            0.369 |
| mixed     | encodec_24khz |              3   |     6.07 |  2.128 |  0.745 |                 160 |            0.345 |
| mixed     | encodec_24khz |              6   |     8.92 |  2.939 |  0.836 |                  80 |            0.348 |
| mixed     | encodec_24khz |             12   |    11.48 |  3.925 |  0.91  |                  40 |            0.348 |
| mixed     | encodec_24khz |             24   |    12.9  |  4.266 |  0.949 |                  20 |            0.377 |
| mixed     | encodec_48khz |              3   |     7.19 |  2.093 |  0.692 |                1600 |            0.905 |
| mixed     | encodec_48khz |              6   |     9.77 |  2.882 |  0.808 |                 800 |            0.917 |
| mixed     | encodec_48khz |             12   |    12.28 |  3.517 |  0.889 |                 400 |            0.938 |
| mixed     | encodec_48khz |             24   |    14.26 |  4.09  |  0.938 |                 200 |            0.969 |

## Recommendation

**Recommended Setting:** EnCodec 24kHz, 24.0 kbps

**Quality Tier:** BEST AVAILABLE (below minimum)

**Average Metrics:**
- SNR: 9.34 dB
- PESQ: 4.033
- STOI: 0.937
- Compression: 20.00x

**Justification:**
- Warning: No bandwidth setting meets minimum quality thresholds
- 24.0 kbps provides best available quality
- Consider alternative approaches (e.g., Complex FFT, higher bandwidth)

**Implementation Notes:**
- Use `EncodecModel.encodec_model_24khz()` (mono, 24kHz)
- Set target bandwidth: `model.set_target_bandwidth(24.0)`
- Expected encoded shape: ~[8, N_frames] for 24.0 kbps
- Integration: Replace mel spectrogram pipeline with EnCodec codes
