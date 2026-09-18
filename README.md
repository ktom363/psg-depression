# Polysomnography-Based Classification of Clinically Significant Depressive Symptoms Using Explainable Deep Learning

This repository contains the code required to reproduce the core pipeline of the study: the
conversion of raw polysomnography (PSG) recordings into waveform images, and the
training of the two image-based Transformer classifiers (Vision Transformer and
Swin Transformer).

---

## Framework Overview (이미지)



---

## Repository contents (수정필요)

| Path | Description |
|---|---|
| `edf_to_images.py` | EDF → 224 × 224 epoch images. Complete preprocessing and rendering in a single pass; no intermediate files. |
| `requirements.txt` | Dependency ranges for the preprocessing stage. |
| `README_KO.md` | Step-by-step execution notes (Korean, Windows). |
| `<training script for ViT>` | Vision Transformer training and evaluation. |
| `<training script for Swin>` | Swin Transformer training and evaluation. |

---
