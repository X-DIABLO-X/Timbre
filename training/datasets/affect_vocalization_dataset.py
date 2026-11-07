import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


VOCALIZATION_CLASSES = [
    "laughter",
    "sighs",
    "gasps",
    "crying",
    "coughing",
    "screaming",
    "yawning",
]


class PreExtractedEmotion2VecDataset(Dataset):
    """Loads pre-extracted 50Hz emotion2vec+ .npy embeddings with continuous
    SAM scale (1-7) Arousal/Valence/Dominance from MSP-Podcast v2.0 and
    7-class human non-speech vocalization labels from VocalSound & Nonspeech7k.

    Continuous VAD mapping:
        MSP-Podcast SAM 1-7 scale is normalized to [0.0, 1.0]:
        norm_val = (val - 1.0) / 6.0
    """

    def __init__(
        self,
        manifest_jsonl: str,
        max_frames_50hz: int = 250,
        default_embed_dim: int = 1024,
    ):
        self.max_frames = max_frames_50hz
        self.default_embed_dim = default_embed_dim
        self.records: List[Dict] = []
        path = Path(manifest_jsonl)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def normalize_sam_1_to_7(val: float) -> float:
        """Maps MSP-Podcast 1-7 Self-Assessment Manikin scale to [0.0, 1.0]."""
        return max(0.0, min(1.0, (val - 1.0) / 6.0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        npy_path = Path(rec.get("npy_path", ""))

        if npy_path.exists():
            emb = np.load(str(npy_path)).astype(np.float32)  # [T, D]
        else:
            # Fallback zero tensor for testing or unextracted paths
            emb = np.zeros((self.max_frames, self.default_embed_dim), dtype=np.float32)

        if emb.ndim == 1:
            emb = emb[None, :]

        t_len, d_dim = emb.shape
        if t_len >= self.max_frames:
            emb = emb[: self.max_frames]
        else:
            pad = np.zeros((self.max_frames - t_len, d_dim), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=0)

        arousal = self.normalize_sam_1_to_7(float(rec.get("arousal_sam", 4.0)))
        valence = self.normalize_sam_1_to_7(float(rec.get("valence_sam", 4.0)))
        dominance = self.normalize_sam_1_to_7(float(rec.get("dominance_sam", 4.0)))

        vocal_label = int(rec.get("vocalization_class_id", 0))

        return {
            "embeddings_50hz": torch.from_numpy(emb),
            "vad_target": torch.tensor([arousal, valence, dominance], dtype=torch.float32),
            "vocalization_target": torch.tensor(vocal_label, dtype=torch.long),
        }
