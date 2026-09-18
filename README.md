# Polysomnography-Based Classification of Clinically Significant Depressive Symptoms Using Explainable Deep Learning

Reference implementation accompanying the manuscript *"Polysomnograph-Based Classification of
Clinically Significant Depressive Symptoms Using Explainable Deep Learning"* (under review,
*Engineering Applications of Artificial Intelligence*).

This repository contains the code required to reproduce the core pipeline of the study: the
conversion of raw polysomnography (PSG) recordings into fixed-layout waveform images, and the
training and evaluation of the two image-based Transformer classifiers (Vision Transformer and
Swin Transformer).

---

## Overview

Overnight PSG is acquired routinely in sleep clinics and contains multichannel physiological
information that is not used for mental-health assessment in standard practice. In this study,
minimally preprocessed 18-channel PSG recordings are rendered into two-dimensional waveform
images through a fixed, fully specified procedure, and Transformer image classifiers are trained
to classify **clinically significant depressive symptoms** at the participant level.

The task is the classification of self-reported symptom burden (SCL-90-R Depression T-score ≥ 63
internally; BDI-II ≥ 20 externally), **not** the diagnosis of a depressive disorder. The intended
setting is opportunistic symptom-risk assessment among patients already undergoing clinical PSG,
not screening in the general population.

Pipeline in brief:

```
EDF recording
  → 18 fixed channels (AASM-based selection)
  → 6-hour window beginning 15 min after recording onset
  → resampling to 100 Hz
  → Savitzky–Golay smoothing (window 25, order 3)
  → per-channel min–max scaling to [0, 1]
  → 720 non-overlapping 30-second epochs
  → 224 × 224 RGB waveform images
  → ViT / Swin Transformer (per-epoch probability)
  → mean over 720 epochs → participant-level score
```

---

## Repository contents

| Path | Description |
|---|---|
| `edf_to_images.py` | EDF → 224 × 224 epoch images. Complete preprocessing and rendering in a single pass; no intermediate files. |
| `requirements.txt` | Dependency ranges for the preprocessing stage. |
| `README_KO.md` | Step-by-step execution notes (Korean, Windows). |
| `<training script for ViT>` | Vision Transformer training and evaluation. |
| `<training script for Swin>` | Swin Transformer training and evaluation. |

> The two model scripts are to be added; replace the placeholder rows above with the actual
> filenames once they are committed.

**Not included.** Polysomnography recordings, questionnaire scores, and any participant-level
data cannot be shared: they are identifiable physiological data governed by the institutional
review board approvals listed below. Numerical baseline models, interpretability analyses, and
the supplementary ablation experiments reported in the manuscript are outside the scope of this
repository.

---

## Preprocessing

`edf_to_images.py` searches a directory for EDF files and converts each one independently of any
label or group assignment.

```bash
python edf_to_images.py --input /path/to/edf_root --output /path/to/image_root
```

Common options:

| Option | Default | Purpose |
|---|---|---|
| `--pattern` | `*.edf` | Filename pattern to search for (e.g. `Traces.edf`). |
| `--id-mode` | `auto` | Whether the participant identifier is taken from the parent directory or the file stem. |
| `--start-second` | `900` | Onset of the analysis window, in seconds from recording start. |
| `--n-epochs` | `720` | Number of consecutive 30-second epochs to render. |
| `--limit` | – | Process only the first *n* recordings (useful for a trial run). |
| `--overwrite` | off | Re-render images that already exist. |

Output layout:

```
<output>/<patient_id>_image/<patient_id>_T900.png
<output>/<patient_id>_image/<patient_id>_T930.png
...
<output>/preprocessing_manifest.csv
```

The manifest records, for each recording, the original sampling rate, recording duration, number
of images written, and the channels rendered as blank rows.

### Fixed channel layout

Each image contains 18 rows in a fixed order; every row occupies one eighteenth of the image
height and the full width. A channel that is absent from a recording is retained as a blank row so
that the row index always denotes the same modality. This is how the external cohort, in which
the frontal EEG derivations F3 and F4 were unavailable, is handled.

| Row | Channel | Modality |
|---|---|---|
| 1–6 | F3, F4, C3, C4, O1, O2 | EEG |
| 7–8 | EOG Left, EOG Right | EOG |
| 9–11 | Chin, Left Leg, Right Leg | EMG |
| 12 | EKG | ECG |
| 13 | Nasal Pressure | Airflow |
| 14 | Abdomen | Respiratory effort |
| 15 | SpO₂ | Oxygen saturation |
| 16 | Snoring Sensor | Snoring |
| 17–18 | Gravity X, Gravity Y | Body position |

Vendor-specific header variants are resolved through an alias table in the script; recordings
exported by other systems may require extending it. Rendering constants — 6 × 6 in at 100 dpi,
white 0.8-pt traces on black, no axes or frames, bicubic downsampling to 224 × 224 — correspond
to Supplementary Algorithm S1 of the manuscript.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 is recommended. `requirements.txt` lists dependency ranges rather than a verified
lockfile; record the resolved versions with `pip freeze` after a successful run if an exact
environment needs to be reproduced. Model training additionally requires PyTorch with CUDA
support; the reported experiments were run on a single NVIDIA RTX A6000.

---

## Data

| Cohort | Site | Label | Purpose |
|---|---|---|---|
| Development | Korea University Anam Hospital | SCL-90-R Depression T-score | Training, validation, internal test |
| External validation | Chonnam National University Hospital | BDI-II total score | External validation |

The retrospective use of de-identified development data was approved by the Institutional Review
Board of Korea University Anam Hospital with a waiver of informed consent (IRB No. 2025AN0405).
The external cohort was collected prospectively with informed consent (IRB No. CNUH-2025-236).
Neither dataset can be redistributed.

To run this code on other recordings, place EDF files under a single directory — one
subdirectory per participant is the expected layout — and point `--input` at it.

---

## Citation

```bibtex
@article{<citation-key>,
  title   = {Polysomnograph-Based Classification of Clinically Significant Depressive Symptoms
             Using Explainable Deep Learning},
  author  = {<authors>},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {<year>},
  doi     = {<doi>}
}
```

> Fill in once the manuscript is accepted, and add the Zenodo archive DOI for this repository.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

Questions about the code may be raised as a GitHub issue. Correspondence regarding the study
should be directed to the corresponding author of the manuscript.
