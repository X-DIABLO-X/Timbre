# Curated Training Datasets Specification & PyTorch Loaders (`datasets.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 3.3)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Potato Speech Emotion Annotation Guide](https://www.potatoannotator.com/docs/guides/speech-emotion-annotation), [MSP-Podcast IEEE 2026](https://www.computer.org/csdl/journal/ta/2026/03/11457331/2fhuqwy7Xji), [WhiStress (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [Nonspeech7k IET](https://digital-library.theiet.org/doi/full/10.1049/sil2.12233)

---

## 1. Complete Training & Calibration Datasets Specification

*(Groundedness Note on Corpus Duration Figures: Primary documentation establishes IEMOCAP at `~12 hours` and RAVDESS / CREMA-D at `7,356` and `7,442` short clips ([Potato Annotation Guide](https://www.potatoannotator.com/docs/guides/speech-emotion-annotation)); a secondary survey paper ([ResearchGate 388880242](https://www.researchgate.net/publication/388880242_Hearing_emotions_fine-tuning_speech_emotion_recognition_models)) anomalously prints `288–310 hours` for those datasets, which we explicitly disregard in favor of the primary corpus specifications).*

| Dataset Name | Domain / Type | Utterances / Clips | Duration & Speakers | Annotations & Specific Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **MSP-Podcast (v2.0)** | Naturalistic Podcast Conversations | **264,705 turns** (`2.75s` to `11s` each) ([Potato](https://www.potatoannotator.com/docs/guides/speech-emotion-annotation)) | **409 hours** (hundreds of podcast speakers) | Primary + 16 secondary emotions + continuous **Valence, Arousal, Dominance (`1–7` SAM scale)** from $\ge 5$ raters. Primary naturalistic training corpus for `Emotion2VecMultiTaskHead`. |
| **MELD** | Multi-Party TV Dialogue (*Friends*) | **13,708 utterances** (`1,433` dialogues) ([Vocalizing Affect](https://vnsguj.ac.in/download/October%20to%20December%202025/Vocalizing%20Affect%20A%20Comprehensive.pdf)) | `~13,000` clips (multi-speaker TV cast) | 7 discrete emotions (*anger, disgust, sadness, joy, neutral, surprise, fear*) + 3 sentiments ([arXiv:2312.06270v2](https://arxiv.org/html/2312.06270v2)). Calibrates fast sitcom/cinema turn-taking. |
| **IEMOCAP** | Dyadic Scripted & Improvised Acting | **9,953 turns** | **~12 hours** (`10` actors: 5M, 5F) ([Potato](https://www.potatoannotator.com/docs/guides/speech-emotion-annotation)) | 8–10 discrete emotions (3 raters) + continuous **Valence, Activation, Dominance (`1–5` scale)**. Standard benchmark where `emotion2vec+` achieves SOTA using linear layers only ([emotion2vec](https://github.com/ddlBoJack/emotion2vec/blob/main/README.md)). |
| **CREMA-D** | Multi-Ethnic Acted Sentences | **7,442 clips** (`12` fixed sentences) | **91 actors** (48M, 43F, ages 20–74) ([Potato](https://www.potatoannotator.com/docs/guides/speech-emotion-annotation)) | 6–7 emotions across **4 intensity levels** (*low, medium, high, unspecified*) + **`0–100` continuous intensity slider** (`~9.8` raters/clip). Used strictly to calibrate prosodic intensity scaling and Layer-9 probing ([WhiStress](https://arxiv.org/html/2505.19103v1)). |
| **RAVDESS** | Professional Acted Speech & Song | **7,356 files** (`1,440` speech WAVs) | **24 actors** (12M, 12F, `24.8 GB` media) ([Potato](https://www.potatoannotator.com/docs/guides/speech-emotion-annotation)) | 8 emotions across **2 calibrated intensity levels** (*normal, strong*) + `1–5` intensity/genuineness scales. Used for whisper-to-shout Crest calibration. |
| **TinyStress-15K & BU-RNC** | Synthetic SSML + Broadcast News | **16,000 samples** (`15K` train + `1K` test) + Broadcast corpus ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)) | **~15 hrs** (TinyStress) + **3–11 hrs** (`6` broadcasters: 3M, 3F, BU-RNC) | Word-level binary sentence stress (`slprl/TinyStress-15K`) and ToBI pitch accent markers (`L*`, `!H*`, `H*`, `L+H*` in BU-RNC) for training `WhiStressLayer9Head`. |
| **VocalSound & Nonspeech7k** | Human Non-Speech Vocalizations | **21,024 clips** (VocalSound) + **7,014 clips** (Nonspeech7k) ([Nonspeech7k](https://www.researchgate.net/publication/371691211_Nonspeech7k_dataset_Classification_and_analysis_of_human_non-speech_sound)) | Mean `4.18s`/clip (`3,365` subjects) + **6.75 hrs** (Nonspeech7k) ([IET](https://digital-library.theiet.org/doi/full/10.1049/sil2.12233)) | **VocalSound** (6 classes: *laughter, sighs, coughs, throat clearing, sneezes, sniffs*, `90.5%` baseline accuracy) + **Nonspeech7k** (7 classes adding *crying, screaming, yawning, breathing*). |

---

## 2. PyTorch Dataset Loader for Task 1 (`TinyStress-15K` & `BU-RNC`) — `training/datasets/tinystress_burnc_dataset.py`

Write the following complete code to `training/datasets/tinystress_burnc_dataset.py`:

```python
import string
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import WhisperProcessor


PUNCTUATION_SET = set(string.punctuation)


class TinyStressAndBuRncDataset(Dataset):
    # Loads slprl/TinyStress-15K from HuggingFace Datasets (or BU-RNC local manifest)
    # and aligns word-level binary stress labels to Whisper BPE tokens.
    # Rule: Non-word punctuation tokens are explicitly supervised as unstressed (0.0).
    def __init__(
        self,
        split: str = "train",
        backbone_id: str = "openai/whisper-small.en",
        max_samples: int | None = None,
    ):
        self.processor = WhisperProcessor.from_pretrained(backbone_id)
        self.ds = load_dataset("slprl/TinyStress-15K", split=split)
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def __len__(self) -> int:
        return len(self.ds)

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
        item = self.ds[idx]
        audio_arr = np.asarray(item["audio"]["array"], dtype=np.float32)
        sr = int(item["audio"]["sampling_rate"])

        features = self.processor.feature_extractor(
            audio_arr, sampling_rate=sr, return_tensors="pt"
        ).input_features[0]

        words = item.get("words", item.get("text", "").split())
        stress_flags = item.get("stress_labels", [0] * len(words))
        token_ids, token_labels = self._align_word_stress_to_bpe(words, stress_flags)

        return {
            "input_features": features,
            "decoder_input_ids": torch.tensor(token_ids, dtype=torch.long),
            "stress_labels": torch.tensor(token_labels, dtype=torch.float32),
        }
```

---

## 3. PyTorch Dataset Loader for Task 2 (`MSP-Podcast v2.0`, `VocalSound`, `Nonspeech7k`) — `training/datasets/affect_vocalization_dataset.py`

Write the following complete code to `training/datasets/affect_vocalization_dataset.py`:

```python
import json
from pathlib import Path
from typing import Dict, List
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
    # Loads pre-extracted 50Hz emotion2vec+ .npy embeddings with continuous
    # SAM scale (1-7) Arousal/Valence/Dominance from MSP-Podcast v2.0 and
    # 7-class human non-speech vocalization labels from VocalSound & Nonspeech7k.
    def __init__(self, manifest_jsonl: str, max_frames_50hz: int = 250):
        self.max_frames = max_frames_50hz
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
        # Maps MSP-Podcast 1-7 Self-Assessment Manikin scale to [0.0, 1.0]
        return max(0.0, min(1.0, (val - 1.0) / 6.0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        emb = np.load(rec["npy_path"]).astype(np.float32)  # [T, D]
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
```
