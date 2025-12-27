The csv should contain the following cols.
1. id
2. file_path
3. emotion
---

Training — Project overview

This folder contains the training scripts and configuration for an emotion
classification model built on top of SpeechBrain. The code implements a
minimal pipeline that reads CSV annotations, prepares datasets, computes
features, builds an ECAPA-based classifier, and runs training using a
SpeechBrain `Brain`.

## Files in this folder

- `train.py` — Main training script.
	- Parses/merges hyperparameters and CLI overrides.
	- Prepares datasets via `datasets.prepare_dataset()`.
	- Provides a fallback `compute_features` if not present in the hparams.
	- Instantiates the model (`models.ECAPAEmotionClassifier`) and a
		`Brain` (`EmotionBrain`).
	- Implements `EmotionBrain.compute_forward`, `compute_objectives`, and
		`on_stage_end`.
	- Includes runtime fallbacks for hparams normalization, CUDA fallback,
		Checkpointer compatibility, PaddedData unwrapping, and a simple
		accuracy accumulator for quick tests.

- `models.py` — Model definitions.
	- `ECAPAEmotionClassifier`: wraps SpeechBrain's `ECAPA_TDNN` encoder and a
		linear classification head. `forward(features)` -> logits.

- `datasets.py` — Data pipeline.
	- `prepare_dataset(csv_path)`: builds a `DynamicItemDataset` from a CSV,
		adds a `CategoricalEncoder` for label encoding, defines an audio loading
		dynamic item (`read_audio`), and sets output keys `['id','signal','label_encoded']`.

- `hparams.yaml` — Hyperparameters and configuration. Typical entries:
	`sample_rate`, `n_mels`, `model` (feat_dim, num_classes), `optimizer`,
	`training` (batch_size, epochs). Also add `train_csv`, `valid_csv`,
	`checkpoint_dir`, and `save_model_path` for running training.

## CSV format

Each CSV should contain at minimum these columns:

- `ID` — unique example identifier
- `file_path` — absolute or relative path to the WAV file
- `emotion` — integer label

Example row:

```
1,"../archive/Actor_01/03-01-01-01-01-01-01.wav",1
```

## Quick start

Install dependencies in your virtual environment:

```bash
pip install speechbrain torch
pip install hyperpyyaml  # optional, for !import in YAML
pip install torchaudio   # optional, improves feature extraction
```

Run training from the repository root:

```bash
python training/train.py hparams.yaml
```

## Notes

- If your YAML uses `!import` tags, install `hyperpyyaml`.
- If `hparams` requests CUDA but your PyTorch build lacks CUDA, the script
	will fall back to CPU and print a warning.
- For production runs, replace the quick fallbacks (e.g., `SimpleAccuracy`)
	with robust SpeechBrain metric objects and pin package versions.
