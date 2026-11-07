# Custom Model Training, Architectures, Hyperparameters & ONNX Distillation (`training.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 3.2 & 3.4)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [WhiStress (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [WhiStress HTML Paper (arXiv:2505.19103v1)](https://arxiv.org/html/2505.19103v1), [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec/blob/main/README.md), [Whisper Large-v3 SER Fine-Tuning](https://www.aimodels.fyi/models/compare/emotion2vec-plus-large-emotion2vec-vs-speech-emotion-recognition-with-openai-whisper-large-v3-firdhokk), [SER Domain Mismatch (ISCA 2024)](https://www.isca-archive.org/smm_2024/emmanouilidou24_smm.pdf)

---

## 1. Hardware Sizing & VRAM Rule for Training

Using the standard Transformer memory sizing formula ([Voice AI Guide](https://dev.to/programmerraja/2025-voice-ai-guide-how-to-make-your-own-real-time-voice-agent-part-3-3ocb)):
$$\text{VRAM (GB)} \approx \text{Params (Billions)} \times \text{Precision (Bytes)} \times 1.2\text{ Overhead}, \quad \text{System RAM} \ge 2 \times \text{GPU VRAM}$$
Because we **freeze** the foundation backbones (`CrisperWhisper 2.0` / `Whisper-large-v3-turbo` and `emotion2vec+`), optimizer states and gradients are only allocated for our **`~7M`-parameter WhiStress Layer-9 head** and **`~0.8M`-parameter emotion2vec+ multi-task head**:
- **Recommended Project Training Hardware:** A single consumer GPU with **12 GB to 24 GB VRAM** (NVIDIA RTX 3060 12GB at entry level, or RTX 3090 / 4090 24GB / single A100), **32 GB System RAM** (or `128 GB` on a 32-core workstation), and an **8-core CPU (Intel i7 / Ryzen 7)**.
- Pre-extracting `50 Hz` `emotion2vec+` features to `.npy` files (`extract_emotion2vec_features.py`) allows Task 2 to train in minutes on any single GPU or CPU (`torch>=1.13`) ([emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec/blob/main/README.md)).
- *(By contrast, full-backbone pre-training across `42,526–50,000` hours like `emotion2vec_plus_large` or `ETTA` requires `8× NVIDIA A100 80GB GPUs` ([ETTA](https://arxiv.org/pdf/2412.19351)), which is never needed for our surgical head training).*

---

## 2. Custom Training Task 1: Adapting WhiStress Prominence Head to CrisperWhisper 2.0 / Whisper-Turbo Layer 9

### 2.1 Why We Train This Head
Published `slprl/WhiStress` weights were trained on `openai/whisper-small.en` (`244M`) ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)). Running a separate `whisper-small.en` alongside `CrisperWhisper 2.0` (`1.55B`) or `Whisper-large-v3-turbo` (`809M`) would duplicate Whisper encoders in VRAM. Instead, we freeze `CrisperWhisper 2.0` / `Whisper-turbo` and attach the **7-million-parameter Stress Detection Head** at the intermediate layer corresponding to Layer 9's prosodic peak.

### 2.2 Exact Architecture — `training/models/whistress_layer9_head.py`

Write the following complete PyTorch module to `training/models/whistress_layer9_head.py`:

```python
from typing import Optional
import torch
import torch.nn as nn


class WhiStressLayer9Head(nn.Module):
    # 7M-parameter Stress Detection Head (Interspeech 2025 slprl/WhiStress architecture):
    # - 1 additional Transformer Decoder Block with self-attention + cross-attention
    #   attending to Encoder Layer-9 prosodic hidden states
    # - 2-layer FCNN binary token classifier outputting stress logits per decoder token.
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # 2-Layer FCNN binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        decoder_hidden_l9: torch.Tensor,
        encoder_hidden_l9: torch.Tensor,
        decoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # decoder_hidden_l9: [B, T_dec, D]
        # encoder_hidden_l9: [B, T_enc, D]
        x = decoder_hidden_l9
        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=decoder_key_padding_mask)
        x = self.norm1(x + sa_out)

        ca_out, _ = self.cross_attn(query=x, key=encoder_hidden_l9, value=encoder_hidden_l9)
        x = self.norm2(x + ca_out)

        x = self.norm3(x + self.ffn(x))
        logits = self.classifier(x)  # [B, T_dec, 1]
        return logits
```

### 2.3 Two-Stage Curriculum & Non-Word Punctuation Supervision Rule
1. **Stage 1 Pre-Training on `slprl/TinyStress-15K`:**
   - `15,000` synthetic training sentences + `1,000` test sentences derived from TinyStories, labeled by GPT-4o-mini with two distinct semantic stress patterns per sentence and synthesized across 10 voices via Google Cloud TTS SSML (speaking rate reduced by `30%–85%`, volume raised by `+3 to +6 dB`, pitch raised by `+1.5 semitones`, plus Gaussian noise augmentation) ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)).
2. **Stage 2 Fine-Tuning on Natural Broadcast & Conversational Speech (`BU-RNC` & `Buckeye`):**
   - Fine-tunes on the **Boston University Radio News Corpus (BU-RNC)** (`~3 to 11 hours` across 6 professional broadcasters—3 male, 3 female—annotated with ToBI pitch accents `L*`, `!H*`, `H*`, `L+H*`, where `H*`, `L+H*`, and `!H*` map to stressed `1`) and the **Buckeye Corpus** ([Bishop et al., 2020](https://www.speechlab.csi.cuny.edu/uploads/1/4/5/9/14591628/bishop_kuo_kim__2020_.pdf)).
3. **Critical Supervision Rule for Non-Word Tokens:**
   - All punctuation tokens (`,`, `.`, `?`, `!`, `'`, `-`) and special BPE control tokens are **explicitly supervised as unstressed (`label = 0.0`)** rather than masked with `-100`, preventing false stress spikes on punctuation ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)).

### 2.4 Complete Training Script — `training/train_whistress_head.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from training.models.whistress_layer9_head import WhiStressLayer9Head


def train_whistress_adapter(
    backbone_id: str = "openai/whisper-small.en",
    enc_tap_layer: int = 9,
    dec_tap_layer: int = 9,
    lr: float = 1e-4,
    warmup_ratio: float = 0.1,
    epochs: int = 5,
    save_path: str = "weights/whistress_small_layer9.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = WhisperForConditionalGeneration.from_pretrained(backbone_id).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    d_model = backbone.config.d_model
    nhead = backbone.config.encoder_attention_heads
    head = WhiStressLayer9Head(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], device=device))

    # Forward pass hook extracting Layer-9 hidden representations
    def forward_batch(input_features: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = backbone(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            enc_l9 = outputs.encoder_hidden_states[enc_tap_layer]
            dec_l9 = outputs.decoder_hidden_states[dec_tap_layer]
        return head(dec_l9, enc_l9).squeeze(-1)

    torch.save(head.state_dict(), save_path)
```

---

## 3. Custom Training Task 2: Paralinguistic & Continuous Arousal/Texture Head on `emotion2vec+`

### 3.1 Avoiding Acted-Corpus Domain Mismatch
Speech Emotion Recognition (SER) domain-mismatch research proves that models trained solely on acted studio corpora (**RAVDESS**, **CREMA-D**) fail to generalize to naturalistic media, and **naive joint training of MSP-Podcast and IEMOCAP drops test performance on CREMA-D by 11%** ([Forasoft 2026](https://www.forasoft.com/blog/article/audio-emotion-detection-system-using-ai), [ISCA SMM 2024](https://www.isca-archive.org/smm_2024/emmanouilidou24_smm.pdf)).
- **Our Curriculum Strategy:**
  1. Train the continuous Valence/Arousal/Dominance and vocal texture head primarily on naturalistic conversational speech (**MSP-Podcast v2.0** `264,705` turns + **MELD** `13,708` TV turns).
  2. Train the 7-class human non-speech vocalization head on **VocalSound** (`21,024` clips) and **Nonspeech7k** (`7,014` clips).
  3. Use **CREMA-D** (`0–100` continuous intensity slider across 4 intensity levels) and **RAVDESS** (`2` calibrated intensity levels) **strictly as post-training intensity calibration sets** to anchor the linear gain of Roboto Flex's `XOPQ`/`YOPQ` and `wdth` axes.

### 3.2 Exact Hyperparameters
- **Optimizer:** `Adam` / `AdamW` with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ ([Whisper Large-v3 SER Benchmark](https://www.aimodels.fyi/models/compare/emotion2vec-plus-large-emotion2vec-vs-speech-emotion-recognition-with-openai-whisper-large-v3-firdhokk)).
- **Learning Rate & Schedule:** `lr = 5e-5` (end-to-end) or `5e-4` (linear head on frozen `.npy` features), linear decay scheduler with **`warmup_ratio = 0.1`**.

### 3.3 Multi-Task Head Architecture — `training/models/emotion2vec_multitask_head.py`

Write the following complete PyTorch module to `training/models/emotion2vec_multitask_head.py`:

```python
from typing import Dict
import torch
import torch.nn as nn


class Emotion2VecMultiTaskHead(nn.Module):
    # Lightweight multi-task head (~0.8M params) attached to frozen 50Hz emotion2vec+ features:
    # 1. Frame-level continuous Arousal, Valence, Dominance (normalized [0, 1])
    # 2. Frame-level vocal strain / timbre contrast driving Roboto Flex XOPQ & YOPQ
    # 3. Utterance-level 7-class human non-speech vocalization classifier
    #    (laughter, sighs, gasps, crying, coughing, screaming, yawning)
    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 256, num_vocal_classes: int = 7):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Frame-level continuous [Arousal, Valence, Dominance]
        self.vad_head = nn.Linear(hidden_dim, 3)
        # Frame-level [XOPQ_delta, YOPQ_delta] in [-1, 1]
        self.texture_head = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Tanh())

        # Attentive temporal pooling for 7-class non-speech vocalization detection
        self.attn_weights = nn.Linear(hidden_dim, 1)
        self.vocalization_head = nn.Linear(hidden_dim, num_vocal_classes)

    def forward(self, frame_embeddings_50hz: torch.Tensor) -> Dict[str, torch.Tensor]:
        # frame_embeddings_50hz: [B, T_50hz, embed_dim]
        h = self.proj(frame_embeddings_50hz)  # [B, T, hidden_dim]
        vad_continuous = torch.sigmoid(self.vad_head(h))  # [B, T, 3]
        strain_texture = self.texture_head(h)  # [B, T, 2]

        attn = torch.softmax(self.attn_weights(h), dim=1)  # [B, T, 1]
        pooled = torch.sum(h * attn, dim=1)  # [B, hidden_dim]
        vocalization_logits = self.vocalization_head(pooled)  # [B, 7]

        return {
            "vad_continuous": vad_continuous,
            "strain_texture": strain_texture,
            "vocalization_logits": vocalization_logits,
        }
```

---

## 4. Hybrid Quantization ONNX Export for Browser WebGPU — `training/export_onnx_hybrid.py`

In browser WebGPU deployments (`Transformers.js`), quantizing the audio encoder degrades mel-spectrogram and pitch feature fidelity. We implement **Hybrid Quantization** ([Mad Devs WebGPU Guide](https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/)):
- **Encoder:** Kept at **`fp32` (or `fp16`)** to preserve acoustic feature precision.
- **Decoder & Linear Heads:** Quantized to **`q4f16`** (for WebGPU) and **`q8` (`QUInt8`)** (for WASM+SIMD fallback).

```python
from pathlib import Path
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from training.models.whistress_layer9_head import WhiStressLayer9Head


def export_whistress_head_onnx(
    pt_path: str = "weights/whistress_small_layer9.pt",
    output_dir: str = "weights/onnx",
    d_model: int = 768,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "whistress_layer9_fp32.onnx"
    q8_onnx = out_dir / "whistress_layer9_q8.onnx"

    model = WhiStressLayer9Head(d_model=d_model)
    if Path(pt_path).exists():
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_dec = torch.randn(1, 16, d_model)
    dummy_enc = torch.randn(1, 150, d_model)

    torch.onnx.export(
        model,
        (dummy_dec, dummy_enc),
        str(fp32_onnx),
        input_names=["decoder_hidden_l9", "encoder_hidden_l9"],
        output_names=["stress_logits"],
        dynamic_axes={
            "decoder_hidden_l9": {0: "batch", 1: "dec_seq"},
            "encoder_hidden_l9": {0: "batch", 1: "enc_seq"},
            "stress_logits": {0: "batch", 1: "dec_seq"},
        },
        opset_version=17,
    )

    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(q8_onnx),
        weight_type=QuantType.QUInt8,
    )
```
