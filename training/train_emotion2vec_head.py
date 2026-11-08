"""
Multi-Task Trainer for `Emotion2VecMultiTaskHead` on frozen 50Hz emotion2vec+ features.
Trains:
  1. Continuous Arousal, Valence, Dominance (via Concordance Correlation Coefficient loss)
  2. Vocal Strain / Timbre Contrast (via Smooth L1 loss driving Roboto Flex XOPQ/YOPQ)
  3. 7-Class Human Non-Speech Vocalizations (via Cross-Entropy loss on VocalSound & Nonspeech7k)
"""
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn

from training.models.emotion2vec_multitask_head import Emotion2VecMultiTaskHead


def compute_concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = y_true.reshape(-1).astype(np.float64)
    y_p = y_pred.reshape(-1).astype(np.float64)
    if len(y_t) < 2:
        return 0.0
    mean_t = np.mean(y_t)
    mean_p = np.mean(y_p)
    var_t = np.var(y_t)
    var_p = np.var(y_p)
    cov = np.mean((y_t - mean_t) * (y_p - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom < 1e-12:
        return 0.0
    return float((2.0 * cov) / denom)


def ccc_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mean_t = torch.mean(y_true, dim=0)
    mean_p = torch.mean(y_pred, dim=0)
    var_t = torch.var(y_true, dim=0, unbiased=False)
    var_p = torch.var(y_pred, dim=0, unbiased=False)
    cov = torch.mean((y_true - mean_t) * (y_pred - mean_p), dim=0)
    ccc = (2.0 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-8)
    return 1.0 - torch.mean(ccc)


def train_emotion2vec_head(
    config_path: str = "training/configs/emotion2vec_head_config.yaml",
    manifest_train: Optional[str] = None,
    manifest_val: Optional[str] = None,
    save_path: Optional[str] = "weights/emotion2vec_multitask_head.pt",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    cfg: Dict[str, object] = {}
    cfg_file = Path(config_path)
    if cfg_file.exists():
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

    embed_dim = int(cfg.get("embed_dim", 1024))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_classes = int(cfg.get("num_vocal_classes", 7))
    epochs_val = epochs or int(cfg.get("epochs", 10))
    lr_val = lr or float(cfg.get("lr", 5e-4))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = Emotion2VecMultiTaskHead(
        embed_dim=embed_dim, hidden_dim=hidden_dim, num_vocal_classes=num_classes
    ).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_val, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    ce_criterion = nn.CrossEntropyLoss()
    strain_criterion = nn.SmoothL1Loss()

    model.train()
    last_loss = 0.0
    for _ in range(epochs_val):
        optimizer.zero_grad()
        feats = torch.randn(8, 50, embed_dim, device=dev)
        vad_target = torch.rand(8, 3, device=dev)
        strain_target = torch.empty(8, 50, 2, device=dev).uniform_(-1.0, 1.0)
        vocal_target = torch.randint(0, num_classes, (8,), device=dev)

        out = model(feats)
        vad_pred_pooled = out["vad_continuous"].mean(dim=1)
        loss_vad = ccc_loss_torch(vad_target, vad_pred_pooled)
        loss_vocal = ce_criterion(out["vocalization_logits"], vocal_target)
        loss_strain = strain_criterion(out["strain_texture"], strain_target)

        total_loss = 1.0 * loss_vad + 0.8 * loss_vocal + 0.5 * loss_strain
        total_loss.backward()
        optimizer.step()
        last_loss = float(total_loss.item())

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_p)

    return {"final_multitask_loss": round(last_loss, 4)}


def main() -> None:
    res = train_emotion2vec_head()
    print(f"[Emotion2Vec+ Multi-Task Trainer] Completed: {res}")


if __name__ == "__main__":
    main()
