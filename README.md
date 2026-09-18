# Polysomnography-Based Classification of Clinically Significant Depressive Symptoms Using Explainable Deep Learning

This repository contains the code required to reproduce the core pipeline of the study: the
conversion of raw polysomnography (PSG) recordings into waveform images, and the
training of the two image-based Transformer classifiers (Vision Transformer and
Swin Transformer).

---

## Framework Overview

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
