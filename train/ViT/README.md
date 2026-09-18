# Polysomnography-Based Classification of Clinically Significant Depressive Symptoms Using Explainable Deep Learning

---

## Vision Transformer
(ViT 사진)

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

### What the script does

Each participant is a single training example. Their epoch images are pushed through the network
in sub-batches of 64, the per-image outputs are pooled into one participant-level logit, and the
loss is computed against the participant label — so one optimiser step corresponds to one
participant. Participants are split 60:20:20 at the participant level with stratification, with
no participant in more than one subset. Training runs for a fixed number of epochs without early
termination, and the checkpoint with the lowest validation loss is retained. At inference the
per-image sigmoid probabilities are averaged, and a mean probability ≥ 0.5 is classified as
clinically significant depressive symptoms.

### Options

| Option | Default | Purpose |
|---|---|---|
| `--model-name` | `vit_base_patch16_224` | timm model identifier. |
| `--checkpoint` | – | Local pretrained state dict; omit to let timm download the weights. |
| `--full-finetune` | off | Update the whole backbone. By default **only the classification head is trained** and the pretrained encoder stays frozen. |
| `--epochs` | `100` | Training epochs; no early stopping. |
| `--lr` / `--weight-decay` | `1e-4` / `1e-4` | AdamW settings, with cosine annealing over the run. |
| `--image-batch-size` | `64` | Images per forward pass within a participant. |
| `--loss` | `bce` | `bce`, `weighted_bce`, or `focal`. |
| `--train-aggregation` | `logit` | Pooling used during training. Inference always averages probabilities. |
| `--downsample-majority` | off | Randomly reduce the majority class to the minority class size before splitting. |
| `--channel-dropout-prob` | `0.0` | Probability of masking one to three channel rows for a participant during training. |
| `--seed` / `--split-seed` | `42` / `1122` | Training seed and participant-split random state. |
| `--min-images` / `--max-images` | `1` / – | Bounds on the epoch images used per participant. |

### Outputs

```
runs/vit/best_vit.pth              lowest-validation-loss checkpoint
runs/vit/training_history.csv      per-epoch loss, accuracy, learning rate
runs/vit/split.json                participant identifiers per subset
runs/vit/test_predictions.csv      participant-level probabilities on the internal test set
runs/vit/run_config.json           resolved settings, parameter counts, library versions
```

`run_config.json` records the trainable and total parameter counts of the resolved configuration,
which is the figure to report alongside the training settings.
