import json
from pathlib import Path
import string
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from transformers import WhisperProcessor
except ImportError:
    WhisperProcessor = None


PUNCTUATION_SET = set(string.punctuation)


class TinyStressAndBuRncDataset(Dataset):
    """Loads slprl/TinyStress-15K from HuggingFace Datasets (or BU-RNC / Buckeye local manifest)
    and aligns word-level binary stress labels to Whisper BPE tokens.

    Explicit Supervision Rule:
        Non-word punctuation tokens (.,!?' etc.) and special tokens are explicitly
        supervised as unstressed (0.0), rather than masked with -100, preventing
        false stress spikes on punctuation.
    """

    def __init__(
        self,
        split: str = "train",
        backbone_id: str = "openai/whisper-small.en",
        max_samples: Optional[int] = None,
        manifest_path: Optional[str] = None,
        dataset_name: str = "slprl/TinyStress-15K",
    ):
        if WhisperProcessor is None:
            raise ImportError(
                "The 'transformers' package is required for TinyStressAndBuRncDataset. "
                "Install it using: pip install transformers"
            )

        self.processor = WhisperProcessor.from_pretrained(backbone_id)
        self.manifest_records: Optional[List[Dict]] = None

        if manifest_path is not None and Path(manifest_path).exists():
            # Local manifest mode for BU-RNC or pre-downloaded TinyStress
            self.manifest_records = []
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.manifest_records.append(json.loads(line))
            if max_samples is not None:
                self.manifest_records = self.manifest_records[:max_samples]
            self.ds = None
        else:
            if load_dataset is None:
                raise ImportError(
                    "The 'datasets' package is required to load HuggingFace datasets. "
                    "Install it using: pip install datasets"
                )
            self.ds = load_dataset(dataset_name, split=split)
            if max_samples is not None:
                self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def __len__(self) -> int:
        if self.manifest_records is not None:
            return len(self.manifest_records)
        return len(self.ds) if self.ds is not None else 0

    def _align_word_stress_to_bpe(
        self, words: List[str], word_stress: List[int]
    ) -> Tuple[List[int], List[float]]:
        token_ids: List[int] = []
        token_labels: List[float] = []

        for word, stress_flag in zip(words, word_stress):
            sub_ids = self.processor.tokenizer.encode(" " + word, add_special_tokens=False)
            for tid in sub_ids:
                decoded = self.processor.tokenizer.decode([tid]).strip()
                # Explicit WhiStress rule: non-word punctuation tokens are supervised as 0.0
                if not decoded or all(ch in PUNCTUATION_SET for ch in decoded):
                    token_labels.append(0.0)
                else:
                    token_labels.append(float(stress_flag))
                token_ids.append(tid)

        return token_ids, token_labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.manifest_records is not None:
            item = self.manifest_records[idx]
            # Handle audio file loading from manifest
            audio_path = item.get("audio_path")
            if audio_path and Path(audio_path).exists():
                import soundfile as sf
                audio_arr, sr = sf.read(audio_path, dtype="float32")
                if audio_arr.ndim > 1:
                    audio_arr = np.mean(audio_arr, axis=1)
            else:
                # Synthetic fallback waveform for testing
                sr = 16000
                audio_arr = np.zeros(sr * 2, dtype=np.float32)

            words = item.get("words", item.get("text", "").split())
            stress_flags = item.get("stress_labels", [0] * len(words))
        else:
            item = self.ds[idx]
            audio_arr = np.asarray(item["audio"]["array"], dtype=np.float32)
            sr = int(item["audio"]["sampling_rate"])
            words = item.get("words", item.get("text", "").split())
            stress_flags = item.get("stress_labels", [0] * len(words))

        features = self.processor.feature_extractor(
            audio_arr, sampling_rate=sr, return_tensors="pt"
        ).input_features[0]

        token_ids, token_labels = self._align_word_stress_to_bpe(words, stress_flags)

        return {
            "input_features": features,
            "decoder_input_ids": torch.tensor(token_ids, dtype=torch.long),
            "stress_labels": torch.tensor(token_labels, dtype=torch.float32),
        }
