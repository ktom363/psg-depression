# PSG-to-Image Preprocessing for Depressive Symptom Classification

This document describes the preprocessing component of an image-based polysomnography (PSG) research pipeline for depressive symptom classification. The script converts European Data Format (EDF) recordings into sequences of 224 × 224 RGB images, with one image per non-overlapping 30-second epoch. Each image uses a fixed arrangement of 18 PSG channels.

Preprocessing is performed independently for each recording and does not require diagnostic labels. Model training, participant-level dataset splitting, prediction aggregation, and performance evaluation are outside the scope of this script.

## Preprocessing pipeline

The default processing order is:

1. **Channel selection:** resolve channel aliases and select the available channels from the fixed 18-channel configuration.
2. **Temporal cropping:** select a six-hour window beginning 15 minutes after the start of the EDF recording, corresponding to the interval from 900 to 22,500 seconds, with the endpoint excluded.
3. **Resampling:** resample the selected window to 100 Hz using MNE.
4. **Smoothing:** apply a Savitzky–Golay filter independently to each channel, using a window length of 25 samples and a polynomial order of 3.
5. **Normalization:** apply channel-wise min–max normalization over the selected window.
6. **Epoch segmentation:** divide the preprocessed signals into non-overlapping 30-second epochs, each containing 3,000 samples per channel.
7. **Image generation:** render each epoch as stacked white traces on a black background, convert to RGB, and resize to 224 × 224 pixels using bicubic interpolation.

## Channel layout

Rows are ordered from the top to the bottom of each image.

| Rows | Channels, in order | Signal type |
| --- | --- | --- |
| 1–6 | F3, F4, C3, C4, O1, O2 | EEG |
| 7–8 | EOG Left, EOG Right | EOG |
| 9–11 | Chin, Left Leg, Right Leg | EMG |
| 12 | EKG | ECG |
| 13 | Nasal Pressure | Airflow |
| 14 | Abdomen | Respiratory effort |
| 15 | SpO2 | Oxygen saturation |
| 16 | Snoring Sensor | Snoring |
| 17–18 | Gravity X, Gravity Y | Position-related signals |

Missing channels retain their designated row positions and appear as blank black rows. The remaining channels do not shift.

## Files and input organization

The commands below assume execution from the repository root (`psg-depression`).

| Path | Purpose |
| --- | --- |
| `README.md` | This documentation |
| `preprocessing/edf_to_images.py` | Preprocessing script |
| `preprocessing/requirements.txt` | Preprocessing dependencies |
| `psg-edf/<ID>/Traces.edf` | Locally supplied EDF recording |
| `psg-images/<ID>_image/` | Generated images |
| `psg-images/preprocessing_manifest.csv` | Processing summary |

Use one uniquely named participant folder per recording for the commands shown here. With `--id-mode parent`, the folder name becomes the output identifier. Each participant folder should contain one matching `Traces.edf` file. 

## Installation

Use Python 3.11 in a dedicated environment. The required packages are MNE, NumPy, SciPy, Matplotlib, and Pillow; dependency ranges are provided in `preprocessing/requirements.txt`.

From the repository root, run:

```bash
conda create -n psg-preprocess python=3.11 -y
conda activate psg-preprocess
python -m pip install -r preprocessing/requirements.txt
```

The preprocessing script does not require a GPU.

## Usage

### Process one recording first

```bash
python preprocessing/edf_to_images.py --input "psg-edf" --output "psg-images" --pattern "Traces.edf" --id-mode parent --limit 1
```

`--limit 1` selects the first matching EDF in sorted path order and generates up to 720 images for that recording; it does not restrict the output to one image.

### Process all matching recordings

```bash
python preprocessing/edf_to_images.py --input "psg-edf" --output "psg-images" --pattern "Traces.edf" --id-mode parent
```

The script searches subdirectories recursively. Output directories are created automatically. Existing images with matching filenames are skipped unless `--overwrite` is supplied. Use a new output directory when changing preprocessing settings to avoid mixing outputs from different configurations.

## Outputs and processing summary

Images are named `<ID>_T<start_second>.png`, where the timestamp refers to recording-relative epoch onset. With default settings and a complete window, filenames range from `<ID>_T900.png` to `<ID>_T22470.png` in 30-second increments.

The summary CSV is written when the batch reaches completion and contains:

| Field | Description |
| --- | --- |
| `patient_id` | Identifier derived from the file path |
| `edf_path` | Source EDF path |
| `status` | Processing status: `ok` or `failed` |
| `original_sfreq_hz` | Sampling frequency reported by MNE before explicit resampling; not necessarily each channel's native EDF frequency |
| `duration_s` | Loaded recording duration in seconds |
| `n_epochs_written` | Number of images written or skipped because matching files already existed |
| `n_missing_channels` | Number of absent or unusable channels |
| `missing_channels` | Semicolon-separated channel names |
| `message` | Recorded error information |

