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

The default start time is relative to **recording onset**, not scored sleep onset. The script does not read sleep-stage annotations or estimate sleep onset. The selected interval is a continuous recording window and may contain both sleep and wakefulness.

A complete six-hour window produces 720 images. If the recording ends earlier, only the available complete epochs are saved; the signal is not padded, and a trailing incomplete epoch is discarded. The explicit 100-Hz resampling step occurs after cropping; MNE may additionally align channels with different sampling rates when loading an EDF file.

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

Channel matching ignores capitalization and leading/trailing whitespace. Supported alternative names are defined in `CHANNEL_ALIASES`, including `LOC`/`ROC` for EOG and `ECG` for EKG. Unlisted channel names are not matched automatically; inspect the EDF headers and the alias dictionary when using recordings from another acquisition system.

Missing channels retain their designated row positions and appear as blank black rows. The remaining channels do not shift. A channel with no finite values after preprocessing is also treated as missing.

## Normalization and rendering details

For each participant and channel, finite values in the smoothed analysis window are normalized as follows:

```text
x_normalized = (x - min(x)) / (max(x) - min(x))
```

Constant channels are mapped to zero. Normalization statistics are computed independently within each recording's selected window, rather than across participants or separately for each epoch. When the recording is shorter than six hours, the available selected window determines these statistics.

The renderer uses a 6 × 6-inch Matplotlib figure at 100 dpi, 18 equal-height axes, and a line width of 0.8 points. Axis labels and ticks are hidden. Tight bounding-box saving is enabled by default, followed by bicubic resizing to 224 × 224 pixels.

**Each channel's vertical axis is automatically scaled within each epoch.** Consequently, the displayed trace height does not preserve the relative amplitude of different epochs on a common 0–1 axis, despite the preceding window-level normalization.

For partial signal gaps, non-finite values are temporarily replaced with zero before smoothing, and the original non-finite positions are subsequently restored to NaN. During rendering, non-finite values are replaced with zero. Thus, partial gaps are drawn as zero-valued segments, whereas wholly missing channels remain blank. If smoothing raises an exception, the script logs a warning and falls back to the unsmoothed, zero-filled trace.

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

Use one uniquely named participant folder per recording for the commands shown here. With `--id-mode parent`, the folder name becomes the output identifier. Each participant folder should contain one matching `Traces.edf` file. Other files, such as `Recording.esrc`, are not used.

No intermediate Parquet files are generated. Diagnostic labels, sleep-onset metadata, and clinical tables are not required as inputs.

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

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--input`, `-i` | Required | Root directory containing EDF files |
| `--output`, `-o` | Required | Output directory |
| `--pattern` | `*.edf` | Filename search pattern |
| `--id-mode` | `auto` | Identifier source: `auto`, `parent`, or `stem` |
| `--start-second` | `900` | Analysis start in seconds from recording onset |
| `--n-epochs` | `720` | Maximum number of consecutive epochs; allowed range: 1–720 |
| `--limit` | No limit | Maximum number of recordings to process |
| `--no-recursive` | Disabled | Search only the input directory itself |
| `--overwrite` | Disabled | Regenerate images with matching filenames |
| `--no-tight-bbox` | Disabled | Save the full intermediate canvas before resizing |
| `--manifest` | Output directory / `preprocessing_manifest.csv` | Custom summary CSV path |
| `--log-level` | `INFO` | Logging verbosity |

The compatibility options `--preprocess-scope` and `--minmax-scope` accept only `window`. Smoothing and normalization always follow cropping and resampling. To inspect all arguments, run `python preprocessing/edf_to_images.py --help`.

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

An `ok` status does not guarantee 720 images or the presence of all 18 channels. Review image counts, missing-channel fields, and representative images before downstream analysis. Short recordings with at least one complete epoch can be marked `ok`. Rendering or file-writing exceptions can interrupt a batch before the summary is saved. The current script can also return exit code 0 when only some recordings succeed, so the exit code alone is not a complete quality check.

## Reproducibility and research use

This README documents the accompanying implementation and its defaults. Reproduction of manuscript results additionally requires the matching input data, preprocessing version and options, participant selection, and downstream analysis procedures. Changes to window selection, operation order, aliases, scaling, or rendering can change the generated images.

The dependency file specifies version ranges, not a frozen environment. After verifying a local run, record the installed versions and retain the command used:

```bash
python -m pip freeze > environment_versions.txt
```

Development checks covered Python syntax and synthetic signals using a mock recording interface, including cropping/resampling order, complete-epoch counts, and RGB image dimensions. These checks do not constitute end-to-end validation with actual EDF files or validation on the user's Windows installation.

EDF recordings and participant-derived outputs are user-supplied local data and are not included in this preprocessing package. Processing summaries may contain participant identifiers and local file paths. This component is intended for research preprocessing and does not itself produce a clinical diagnosis.
