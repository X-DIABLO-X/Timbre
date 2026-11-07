"""WhiStress Layer-9 Prominence Head Training Pipeline.

Two-Stage Curriculum:
  Stage 1: Pre-training on slprl/TinyStress-15K synthetic sentence stress corpus.
  Stage 2: Fine-tuning on natural broadcast and conversational speech (BU-RNC & Buckeye).

Enforces:
  1. Frozen Whisper backbone (openai/whisper-small.en, nyralabs/CrisperWhisper2.0_large,
     or openai/whisper-large-v3-turbo). Optimizer states only allocated for ~7M head.
  2. Intermediate extraction from Layer 9 (enc_tap_layer=9, dec_tap_layer=9; or
     enc_tap_layer=24, dec_tap_layer=24/3 on 32-layer Large).
  3. Non-word punctuation supervision rule: all punctuation tokens are explicitly
     supervised with label = 0.0 rather than masked with -100.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import yaml

# Add project root to sys.path if running as a direct script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    WhisperForConditionalGeneration = None
    WhisperProcessor = None
    get_linear_schedule_with_warmup = None

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
except ImportError:
    f1_score = None
    precision_score = None
    recall_score = None

from training.models.whistress_layer9_head import WhiStressLayer9Head
from training.datasets.tinystress_burnc_dataset import TinyStressAndBuRncDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_whistress_head")


def whistress_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates and pads variable-length decoder sequences and stress labels."""
    input_features = torch.stack([item["input_features"] for item in batch], dim=0)

    max_dec_len = max(len(item["decoder_input_ids"]) for item in batch)
    padded_dec_ids = []
    padded_labels = []
    padding_masks = []

    for item in batch:
        dec_ids = item["decoder_input_ids"]
        labels = item["stress_labels"]
        cur_len = len(dec_ids)
        pad_len = max_dec_len - cur_len

        if pad_len > 0:
            padded_dec_ids.append(torch.cat([dec_ids, torch.full((pad_len,), 50257, dtype=torch.long)]))
            # Punctuation / padding supervised as 0.0
            padded_labels.append(torch.cat([labels, torch.zeros(pad_len, dtype=torch.float32)]))
            mask = torch.cat([torch.zeros(cur_len, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
            padding_masks.append(mask)
        else:
            padded_dec_ids.append(dec_ids)
            padded_labels.append(labels)
            padding_masks.append(torch.zeros(cur_len, dtype=torch.bool))

    return {
        "input_features": input_features,
        "decoder_input_ids": torch.stack(padded_dec_ids, dim=0),
        "stress_labels": torch.stack(padded_labels, dim=0),
        "padding_mask": torch.stack(padding_masks, dim=0),
    }


def compute_metrics(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Computes F1, precision, and recall scores."""
    binary_preds = (preds >= threshold).astype(int)
    binary_targets = targets.astype(int)

    if f1_score is not None:
        f1 = float(f1_score(binary_targets, binary_preds, zero_division=0))
        prec = float(precision_score(binary_targets, binary_preds, zero_division=0))
        rec = float(recall_score(binary_targets, binary_preds, zero_division=0))
    else:
        # Fallback pure numpy implementation
        tp = np.sum((binary_preds == 1) & (binary_targets == 1))
        fp = np.sum((binary_preds == 1) & (binary_targets == 0))
        fn = np.sum((binary_preds == 0) & (binary_targets == 1))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    acc = float(np.mean(binary_preds == binary_targets))
    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}


def train_whistress_adapter(
    backbone_id: str = "openai/whisper-small.en",
    enc_tap_layer: int = 9,
    dec_tap_layer: int = 9,
    lr: float = 1e-4,
    warmup_ratio: float = 0.1,
    epochs: int = 5,
    batch_size: int = 16,
    pos_weight: float = 2.5,
    save_path: str = "weights/whistress_small_layer9.pt",
    dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    dry_run: bool = False,
) -> WhiStressLayer9Head:
    """Trains WhiStress prominence head attached to frozen Whisper Layer 9."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing WhiStress training on device: %s", device)

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[Dry Run] Running mock training step without downloading heavy Whisper backbone.")
        d_model = 768
        head = WhiStressLayer9Head(d_model=d_model, nhead=12, dim_feedforward=d_model * 4).to(device)
        dummy_dec = torch.randn(2, 10, d_model, device=device)
        dummy_enc = torch.randn(2, 50, d_model, device=device)
        logits = head(dummy_dec, dummy_enc).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, torch.zeros_like(logits))
        loss.backward()
        torch.save(head.state_dict(), save_path)
        logger.info("[Dry Run] Successfully executed backward pass and saved test weights to %s", save_path)
        return head

    if WhisperForConditionalGeneration is None:
        raise ImportError("transformers is required. Install via: pip install transformers")

    logger.info("Loading backbone model %s (frozen)...", backbone_id)
    backbone = WhisperForConditionalGeneration.from_pretrained(backbone_id).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    d_model = backbone.config.d_model
    nhead = backbone.config.encoder_attention_heads
    head = WhiStressLayer9Head(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    if dataset is None:
        logger.info("Instantiating TinyStressAndBuRncDataset(split='train')...")
        dataset = TinyStressAndBuRncDataset(split="train", backbone_id=backbone_id)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=whistress_collate_fn,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = None

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["stress_labels"].to(device)
            mask = batch["padding_mask"].to(device)

            with torch.no_grad():
                outputs = backbone(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                enc_l9 = outputs.encoder_hidden_states[enc_tap_layer]
                dec_l9 = outputs.decoder_hidden_states[dec_tap_layer]

            logits = head(dec_l9, enc_l9, decoder_key_padding_mask=mask).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        logger.info("Epoch %d/%d - Average Loss: %.4f", epoch, epochs, avg_loss)

    torch.save(head.state_dict(), save_path)
    logger.info("Saved WhiStress checkpoint to %s", save_path)
    return head


def train_stage1_tinystress(
    config_path: str = "training/configs/whistress_layer9_config.yaml",
    save_path: Optional[str] = None,
    dry_run: bool = False,
) -> WhiStressLayer9Head:
    """Stage 1: Pre-training on slprl/TinyStress-15K synthetic sentence stress corpus."""
    logger.info("=== Stage 1 Pre-Training: TinyStress-15K ===")
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    backbone_id = cfg.get("backbone", {}).get("default_id", "openai/whisper-small.en")
    enc_tap = cfg.get("backbone", {}).get("enc_tap_layer", 9)
    dec_tap = cfg.get("backbone", {}).get("dec_tap_layer", 9)
    lr = float(cfg.get("training", {}).get("lr", 1e-4))
    epochs = int(cfg.get("training", {}).get("stage1", {}).get("epochs", 5))
    batch_size = int(cfg.get("training", {}).get("stage1", {}).get("batch_size", 16))
    out_file = save_path or str(Path(cfg.get("paths", {}).get("save_dir", "weights")) / cfg.get("paths", {}).get("checkpoint_file", "whistress_small_layer9.pt"))

    return train_whistress_adapter(
        backbone_id=backbone_id,
        enc_tap_layer=enc_tap,
        dec_tap_layer=dec_tap,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=out_file,
        dry_run=dry_run,
    )


def train_stage2_burnc(
    config_path: str = "training/configs/whistress_layer9_config.yaml",
    checkpoint_in: Optional[str] = None,
    save_path: Optional[str] = None,
    dry_run: bool = False,
) -> WhiStressLayer9Head:
    """Stage 2: Fine-tuning on natural broadcast and conversational speech (BU-RNC / Buckeye)."""
    logger.info("=== Stage 2 Fine-Tuning: BU-RNC & Buckeye Natural Speech ===")
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    backbone_id = cfg.get("backbone", {}).get("default_id", "openai/whisper-small.en")
    enc_tap = cfg.get("backbone", {}).get("enc_tap_layer", 9)
    dec_tap = cfg.get("backbone", {}).get("dec_tap_layer", 9)
    lr = float(cfg.get("training", {}).get("stage2", {}).get("lr", 5e-5))
    epochs = int(cfg.get("training", {}).get("stage2", {}).get("epochs", 3))
    batch_size = int(cfg.get("training", {}).get("stage2", {}).get("batch_size", 16))
    out_file = save_path or str(Path(cfg.get("paths", {}).get("save_dir", "weights")) / "whistress_burnc_layer9.pt")

    return train_whistress_adapter(
        backbone_id=backbone_id,
        enc_tap_layer=enc_tap,
        dec_tap_layer=dec_tap,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=out_file,
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description="WhiStress Layer-9 Prominence Head Trainer")
    parser.add_argument("--stage", type=str, default="1", choices=["1", "2", "both"])
    parser.add_argument("--config", type=str, default="training/configs/whistress_layer9_config.yaml")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Run quick 1-step test without heavy models")
    args = parser.parse_args()

    if args.stage in ["1", "both"]:
        train_stage1_tinystress(args.config, save_path=args.save_path, dry_run=args.dry_run)
    if args.stage in ["2", "both"]:
        train_stage2_burnc(args.config, save_path=args.save_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
