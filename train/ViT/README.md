# Polysomnography-Based Classification of Clinically Significant Depressive Symptoms Using Explainable Deep Learning

---

## Vision Transformer

<img src="../img/img3.png">

---

## Setup

Python 3.11 is recommended. 

With conda or miniconda:

```bash
conda create -n psg-dl python=3.11 -y
conda activate psg-dl
```
Then, inside the activated environment:

```bash
# CUDA 12.1 build, as an example; use the index URL that matches the local driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Vision Transformer training

```bash
python train_vit.py --image-root /path/to/images \
                    --labels labels.csv \
                    --output-dir runs/vit
```

`labels.csv` holds one row per participant:

```csv
patient_id,label
00000001,1
00000002,0
```

`label` is 1 for clinically significant depressive symptoms and 0 otherwise. Participants listed
in the file but absent from the image directory are skipped with a warning. 

### Options

| Option | Default | Purpose |
|---|---|---|
| `--model-name` | `vit_base_patch16_224` | timm model identifier. |
| `--checkpoint` | – | Local pretrained state dict; omit to let timm download the weights. |
| `--head-only` | off | Freeze the pretrained encoder and update the classification head only. By default **the whole backbone is fine-tuned**. |
| `--dropout` | `0.2` | Dropout applied to the pooled representation before the two-class classifier. |
| `--normalization` | `model` | Channel statistics for input normalisation. `model` uses the statistics the pretrained weights of the resolved backbone were trained with; `half` forces 0.5/0.5. |
| `--epochs` | `100` | Training epochs; no early stopping. |
| `--lr` / `--weight-decay` | `1e-4` / `1e-4` | AdamW settings, with cosine annealing over the run. |
| `--batch-size` | `32` | Images per optimiser step. |
| `--eval-batch-size` | `32` | Images per forward pass at validation and test time. |
| `--seed` / `--split-seed` | `42` / `1112` | Training seed and participant-split random state. |
| `--min-images` / `--max-images` | `1` / – | Bounds on the epoch images used per participant. |
| `--test-size` / `--val-size` | `0.2` / `0.2` | Proportions of the 60:20:20 participant split. |
| `--no-pretrained` | off | Initialise the backbone randomly instead of from pretrained weights. |
| `--no-amp` | off | Disable mixed-precision training. |
| `--num-workers` / `--device` | `0` / auto | Dataloader workers and compute device. |

### Outputs

```
runs/vit/best_vit.pth              lowest-validation-loss checkpoint
runs/vit/training_history.csv      per-epoch loss, accuracy, participant-level metrics, learning rate
runs/vit/split.json                participant identifiers per subset
runs/vit/test_predictions.csv      participant-level probabilities on the internal test set
runs/vit/run_config.json           resolved settings, parameter counts, library versions
```

`run_config.json` records the trainable and total parameter counts of the resolved configuration,
which is the figure to report alongside the training settings, together with the loss, decision
threshold, scheduler, normalisation, batch size, image counts per subset, and mixed-precision
state of the run.
