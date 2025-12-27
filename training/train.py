# training/train.py

"""
Main training script for emotion classifier using SpeechBrain.

This script:
  - Loads train/validation CSVs
  - Prepares DynamicItemDatasets using your prepare_dataset()
  - Builds an ECAPA + Dense classifier model
  - Trains using a SpeechBrain Brain class
"""

import sys
import os
import torch
import speechbrain as sb
try:
    from hyperpyyaml import load_hyperpyyaml
except Exception:
    load_hyperpyyaml = None

from datasets import prepare_dataset
from models import ECAPAEmotionClassifier  


def _unwrap_tensor(obj):
    """Extract a torch.Tensor from SpeechBrain PaddedData or similar wrappers.

    Returns the original object if extraction fails.
    """
    if isinstance(obj, torch.Tensor):
        return obj

    for attr in ("data", "padded", "values", "tensor", "_data"):
        val = getattr(obj, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                pass
        if isinstance(val, torch.Tensor):
            return val

    try:
        lst = list(obj)
        if lst and isinstance(lst[0], torch.Tensor):
            return torch.nn.utils.rnn.pad_sequence(lst, batch_first=True)
    except Exception:
        pass

    return obj


class SimpleAccuracy:
    """Lightweight accuracy accumulator used when a proper metric object isn't provided.

    It accepts probability/logit tensors and integer label tensors, accumulates
    correct/total counts across batches, and exposes `log(epoch)` and `report()`.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def __call__(self, predictions, targets):
        # predictions: logits/probs (batch, classes) or (batch,)
        # targets: integer labels (batch,) or (batch,1)
        # Extract tensors if wrapped
        preds = predictions if isinstance(predictions, torch.Tensor) else _unwrap_tensor(predictions)
        labs = targets if isinstance(targets, torch.Tensor) else _unwrap_tensor(targets)

        if preds is None or labs is None:
            return

        # Convert model outputs to label indices
        try:
            if preds.dim() > 1:
                pred_labels = preds.argmax(dim=-1)
            else:
                pred_labels = (preds > 0.5).long()
        except Exception:
            # If preds can't be indexed, bail out
            return

        # Normalize shapes: squeeze last dim if it's singleton, then flatten
        if labs.dim() > 1 and labs.size(-1) == 1:
            labs = labs.squeeze(-1)

        pred_labels = pred_labels.view(-1).long()
        labs = labs.view(-1).long()

        # Truncate to smallest length to avoid broadcasting issues
        minlen = min(pred_labels.numel(), labs.numel())
        if minlen == 0:
            return
        if pred_labels.numel() != labs.numel():
            pred_labels = pred_labels[:minlen]
            labs = labs[:minlen]

        correct = int((pred_labels == labs).sum().item())
        total = int(minlen)
        self.correct += correct
        self.total += total

    def log(self, epoch):
        acc = self.correct / self.total if self.total > 0 else 0.0
        print(f"Validation epoch {epoch} - accuracy: {acc:.4f}")
        self.reset()

    def report(self):
        acc = self.correct / self.total if self.total > 0 else 0.0
        print(f"Final accuracy: {acc:.4f}")

class EmotionBrain(sb.Brain):
    """
    SpeechBrain trainer class for emotion classification.
    """

    def compute_forward(self, batch, stage):
        """Runs the forward pass."""
        batch = batch.to(self.device)

        signals = batch.signal  # waveform from DIP
        labels = batch.label_encoded  # encoded emotion labels

        # Feature extraction defined in hparams
        features = self.hparams.compute_features(signals)

        # Forward through the model
        logits = self.modules.model(features)

        return logits, labels

    def compute_objectives(self, predictions, batch, stage):
        """Computes loss and any metrics."""
        logits, labels = predictions

        # Unwrap SpeechBrain padded wrappers to plain tensors when needed
        logits = _unwrap_tensor(logits)
        labels = _unwrap_tensor(labels)

        # CrossEntropyLoss expects logits and label indices
        loss = self.hparams.loss(logits, labels)

        # Track accuracy for validation/testing
        if stage != sb.Stage.TRAIN:
            self.hparams.accuracy(torch.softmax(logits, dim=-1), labels)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Logs stats per stage."""
        if stage == sb.Stage.VALID:
            self.hparams.accuracy.log(epoch)

        if stage == sb.Stage.TEST:
            self.hparams.accuracy.report()

        super().on_stage_end(stage, stage_loss, epoch)


def main():
    # Load hyperparameters and overrides (from CLI)
    # `sb.parse_arguments()` may return a tuple (e.g., (hparams, run_opts))
    # or a Namespace/dict. Normalize to a plain dict so we can index by string keys.
    _raw_hparams = sb.parse_arguments()

    # If a tuple was returned, try to find a dict-like or namespace inside it.
    if isinstance(_raw_hparams, tuple):
        found = None
        for item in _raw_hparams:
            if isinstance(item, dict):
                found = item
                break
            if hasattr(item, "__dict__"):
                found = item
                break
        _raw_hparams = found if found is not None else _raw_hparams[0]

    # If it's an argparse.Namespace or object, convert to dict
    if not isinstance(_raw_hparams, dict):
        try:
            hparams = vars(_raw_hparams)
        except Exception:
            # Fallback: coerce to dict if possible
            try:
                hparams = dict(_raw_hparams)
            except Exception:
                # As a last resort, keep the original value (will raise later)
                hparams = _raw_hparams
    else:
        hparams = _raw_hparams

    # Validate required hyperparameters to give a clear error message
    required_top_keys = [
        "train_csv",
        "valid_csv",
        "checkpoint_dir",
        "save_model_path",
        "model",
        "optimizer",
        "training",
    ]

    if isinstance(hparams, dict):
        missing = [k for k in required_top_keys if k not in hparams]
        # If keys are missing, try loading the YAML file passed as first CLI arg
        if missing:
            if len(sys.argv) > 1 and sys.argv[1].endswith(('.yml', '.yaml')):
                yaml_path = sys.argv[1]
                if os.path.exists(yaml_path):
                    if load_hyperpyyaml is None:
                        raise ImportError(
                            "hyperpyyaml is required to load hparams from YAML.\n"
                            "Install with: pip install hyperpyyaml"
                        )
                    with open(yaml_path) as fin:
                        loaded = load_hyperpyyaml(fin)
                    # merge loaded hparams into hparams if loaded is dict-like
                    if isinstance(loaded, dict):
                        hparams.update(loaded)
                        missing = [k for k in required_top_keys if k not in hparams]
                    # After merging YAML, try to resolve dotted strings to callables
                    if isinstance(hparams, dict):
                        if "loss" in hparams and isinstance(hparams["loss"], str):
                            try:
                                hparams["loss"] = resolve_dotted(hparams["loss"])
                            except Exception:
                                pass
                        if "accuracy" in hparams and isinstance(hparams["accuracy"], str):
                            try:
                                hparams["accuracy"] = resolve_dotted(hparams["accuracy"])
                            except Exception:
                                pass
            if missing:
                available = list(hparams.keys())
                raise KeyError(
                    f"Missing required hparams keys: {missing}. Available keys: {available}.\n"
                    "Please add these to your hparams YAML or pass overrides via CLI."
                )

    # Resolve dotted import strings for common callable hparams (fallback)
    import importlib

    def resolve_dotted(dotted):
        if not isinstance(dotted, str):
            return dotted
        if "." not in dotted:
            return dotted
        module_name, attr = dotted.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception:
            return dotted

    if isinstance(hparams, dict):
        if "loss" in hparams:
            hparams["loss"] = resolve_dotted(hparams["loss"])
        if "accuracy" in hparams:
            hparams["accuracy"] = resolve_dotted(hparams["accuracy"])

    # If accuracy resolved to a plain function, replace it with `SimpleAccuracy`
    # which provides the expected `.log()` and `.report()` methods.
    if isinstance(hparams, dict) and "accuracy" in hparams:
        acc = hparams["accuracy"]
        if callable(acc) and not (hasattr(acc, "log") and hasattr(acc, "report")):
            hparams["accuracy"] = SimpleAccuracy()

    # Ensure device compatibility: if CUDA requested but not available, fall back to CPU
    if isinstance(hparams, dict):
        dev = hparams.get("device")
        try:
            if isinstance(dev, str) and dev.lower().startswith("cuda") and not torch.cuda.is_available():
                print(f"Warning: CUDA requested ({dev}) but PyTorch has no CUDA. Falling back to 'cpu'.")
                hparams["device"] = "cpu"
        except Exception:
            # ignore any unexpected device format
            pass

    # Provide a fallback compute_features function if not supplied in hparams.
    # Prefer torchaudio's MelSpectrogram when available, otherwise use STFT magnitude.
    if isinstance(hparams, dict) and "compute_features" not in hparams:
        def _extract_tensor(obj):
            # If it's already a tensor
            if isinstance(obj, torch.Tensor):
                return obj

            # Try common attributes that might hold the tensor inside SpeechBrain PaddedData
            for attr in ("data", "padded", "values", "tensor", "_data"):
                val = getattr(obj, attr, None)
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        pass
                if isinstance(val, torch.Tensor):
                    return val

            # Try converting from iterable (list of tensors)
            try:
                lst = list(obj)
                if lst and isinstance(lst[0], torch.Tensor):
                    return torch.nn.utils.rnn.pad_sequence(lst, batch_first=True)
            except Exception:
                pass

            raise TypeError("Cannot extract tensor from signals object of type %s" % type(obj))

        try:
            import torchaudio

            mel_kwargs = {
                "sample_rate": int(hparams.get("sample_rate", 16000)),
                "n_mels": int(hparams.get("n_mels", 80)),
                "n_fft": int(hparams.get("n_fft", 400)),
                "hop_length": int(hparams.get("hop_length", 160)),
                "win_length": int(hparams.get("win_length", 400)),
            }
            mel_transform = torchaudio.transforms.MelSpectrogram(**mel_kwargs)

            def compute_features(signals):
                # Accept SpeechBrain PaddedData or plain tensors
                signals_t = _extract_tensor(signals)
                if signals_t.dim() == 3:
                    signals_t = signals_t.squeeze(1)
                mel = mel_transform(signals_t)
                return torch.log(mel + 1e-6).transpose(1, 2)

        except Exception:
            # fallback: simple STFT magnitude
            n_fft = int(hparams.get("n_fft", 400))
            hop_length = int(hparams.get("hop_length", 160))

            def compute_features(signals):
                signals_t = _extract_tensor(signals)
                if signals_t.dim() == 3:
                    signals_t = signals_t.squeeze(1)
                stft = torch.stft(signals_t, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, return_complex=True)
                mag = torch.abs(stft)
                return mag.transpose(1, 2)

        hparams["compute_features"] = compute_features

    # 👉 STEP 1 — prepare training & validation data
    train_data = prepare_dataset(hparams["train_csv"])
    valid_data = prepare_dataset(hparams["valid_csv"])

    # 👉 STEP 2 — build the Emotion classifier model
    # Ensure model hyperparameters are numeric (resolve refs or strings)
    model_cfg = hparams.get("model", {}) if isinstance(hparams, dict) else {}
    try:
        feat_dim = int(model_cfg.get("feat_dim", hparams.get("n_mels", 80)))
    except Exception:
        feat_dim = int(hparams.get("n_mels", 80))
    try:
        num_classes = int(model_cfg.get("num_classes", hparams.get("num_classes", 2)))
    except Exception:
        num_classes = int(hparams.get("num_classes", 2))

    model = ECAPAEmotionClassifier(
        feat_dim=feat_dim,
        num_classes=num_classes,
    )

    # 👉 STEP 3 — instantiate the SpeechBrain Brain class
    brain = EmotionBrain(
        modules={"model": model},
        hparams=hparams,
        opt_class=lambda params: torch.optim.Adam(
            params, lr=hparams["optimizer"]["lr"]
        ),
        checkpointer=sb.utils.checkpoints.Checkpointer(
            hparams["checkpoint_dir"]
        ),
    )

    # 👉 STEP 4 — train!
    brain.fit(
        epoch_counter=range(hparams["training"]["epochs"]),
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs={"batch_size": hparams["training"]["batch_size"]},
        valid_loader_kwargs={"batch_size": hparams["training"]["batch_size"]},
    )

    # Save the best model
    torch.save(model.state_dict(), hparams["save_model_path"])


if __name__ == "__main__":
    main()
