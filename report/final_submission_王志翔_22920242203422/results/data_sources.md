# Data Sources — Additional Challenge Experiments

## 1. DEMAND (Diverse Environments Multichannel Acoustic Noise Database)

| Field | Value |
|-------|-------|
| Dataset | DEMAND |
| Scene | PCAFETER (Cafeteria) |
| Download URL | https://zenodo.org/records/1227121 |
| File | PCAFETER_16k.zip (107 MB) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation | Thiemann, J., Ito, N., & Vincent, E. (2013). DEMAND: Diverse Environments Multichannel Acoustic Noise Database. |
| Used files | PCAFETER/ch01.wav (channel 1, 16 kHz, 300s) |
| Processing | Resampled to 8 kHz, trimmed to 30s, peak normalized to 0.95 |

## 2. THCHS-30 (Chinese Speech)

| Field | Value |
|-------|-------|
| Dataset | THCHS-30 |
| Download URL | https://www.openslr.org/18/ |
| Files | data_thchs30.tgz (6.1 GB training data) |
| License | Apache 2.0 |
| Citation | Wang, D., & Zhang, X. (2015). THCHS-30: A Free Chinese Speech Corpus. |
| Used files | |
| | `data_thchs30/train/A23_150.wav` — Speaker A23, utterance 150 |
| | `data_thchs30/train/A23_246.wav` — Speaker A23, utterance 246 |
| | `data_thchs30/train/B33_465.wav` — Speaker B33, utterance 465 |
| Processing | Resampled to 8 kHz, trimmed to 4.0s, peak normalized to 0.95 |
| Speaker check | A23_150 and A23_246 are from the same speaker (target/enrollment pair) |
| | B33_465 is from a different speaker (interferer) |

## 3. LibriSpeech (English Speech)

| Field | Value |
|-------|-------|
| Dataset | LibriSpeech ASR Corpus |
| Download URL | https://www.openslr.org/12/ |
| Files | test-clean.tar.gz (331 MB) — already on server |
| License | CC BY 4.0 |
| Citation | Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. |
| Used files | |
| | `test-clean/121/121-*.flac` — Speaker 121, utterance 0 (en_target) |
| | `test-clean/121/121-*.flac` — Speaker 121, utterance 1 (en_enroll) |
| | `test-clean/1089/1089-*.flac` — Speaker 1089, utterance 0 (en_interferer) |
| Processing | Resampled to 8 kHz, trimmed to 4.0s, peak normalized to 0.95 |
| Speaker check | Both en_target and en_enroll from speaker 121 (target/enrollment pair) |
| | en_interferer from speaker 1089 (different speaker) |

## 4. Prepared Files

All files in `experiments/additional_challenge/prepared_wavs/`:

| File | Source | Speaker | Duration | Purpose |
|------|--------|---------|----------|---------|
| zh_target.wav | THCHS-30 A23_150 | A23 | 4.0s | Chinese target speech |
| zh_enroll.wav | THCHS-30 A23_246 | A23 | 4.0s | Chinese enrollment (same speaker) |
| zh_interferer.wav | THCHS-30 B33_465 | B33 | 4.0s | Chinese interferer (different speaker) |
| en_target.wav | LibriSpeech 121-0000 | 121 | 4.0s | English target speech |
| en_enroll.wav | LibriSpeech 121-0001 | 121 | 4.0s | English enrollment (same speaker) |
| en_interferer.wav | LibriSpeech 1089-0000 | 1089 | 4.0s | English interferer (different speaker) |
| noise_cafeteria.wav | DEMAND PCAFETER ch01 | - | 30.0s | Real cafeteria background noise |

All files: 8 kHz, mono, 16-bit PCM WAV, peak normalized to 0.95.
