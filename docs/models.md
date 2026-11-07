# Complete Model Inventory, Checkpoints, Tensor Shapes & Wrappers (`models.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 3)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), [NVIDIA Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), [WhiStress (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [FunAudioLLM SenseVoice](https://fun-audio-llm.github.io/), [emotion2vec+ Large](https://huggingface.co/emotion2vec/emotion2vec_plus_large), [emotion2vec-web-distill](https://huggingface.co/thomashallock/emotion2vec-web-distill), [Moonshine ASR](https://webai.show/models/moonshine-asr/)

---

## 1. Executive Summary: Off-the-Shelf vs. Custom Training Split

Over **95% of the platform's total parameter weight** is loaded directly from pre-trained open-source checkpoints without modification. Custom training is surgically restricted to **two lightweight adapter/head modules (`~7M` to `15M` trainable parameters)** that attach to frozen foundation backbones ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_large)).

*(Groundedness Note: Exact parameter counts and millisecond latency benchmarks for [SenseVoice-Small/Large](https://fun-audio-llm.github.io/) and exact standalone GPU VRAM consumption figures for [Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) are not disclosed in their primary model cards and are documented accordingly below).*

---

## 2. Master Model Inventory & Role Matrix

| Pipeline Stage | Model Name & Exact HuggingFace / Repo ID | Parameters & Disk / Memory Footprint | Action: Off-the-Shelf vs. Train | Tensor I/O & Primary Output in `CaptionSpec` |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0a: VAD Gate** | **Silero VAD** (`snakers4/silero-vad` ONNX / TorchScript JIT) | Lightweight ONNX (`~1.8 MB` disk) | **Off-the-Shelf** | **In:** `[1, 512]` (`32ms` @ `16kHz`) or `10-20ms` window.<br>**Out:** Speech probability $p \in [0, 1]$ + `SILENCE_RMS` gate ([kami-subs](https://github.com/MohammdKopa/kami-subs)). |
| **Stage 0b: Vocal Stem Separation** | **SCNet** (`starrytong/SCNet`) or **HT-Demucs v4** (`htdemucs`) | 4-stem separator (**SCNet uses only 48% of HT-Demucs CPU inference time** ([HuggingFace Papers](https://huggingface.co/papers?q=CPU+inference))) | **Off-the-Shelf** *(Optional VOD)* | **In:** Stereo/Mono waveform.<br>**Out:** Isolated `vocals` stem (routed to Streams 1–3) + `music/other` stem (routed to Stream 4) ([UVR5 Guide](https://ai.miraheze.org/wiki/UVR5)). |
| **Stream 1a: Offline / Buffered ASR & Aligner** | **CrisperWhisper 2.0** (`nyrahealth/CrisperWhisper` / `nyralabs/CrisperWhisper2.0_large` + `turbo` draft on CTranslate2) | `1.55B` (`large`) + `809M` (`turbo`); `~2.9 GB` (INT8) to `4.3–5.8 GB` (FP16) VRAM ([CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)) | **Off-the-Shelf** | **In:** `16kHz` mono `float32` (`<=30s` window with conditional continuation).<br>**Out:** Verbatim text (`[um]`, `[uh]`, false starts) + **29.6 ms TIMIT / 41 ms conversational** word boundaries via Viterbi cross-attention decoding. |
| **Stream 1b: Ultra-Low-Latency Live ASR** | **NVIDIA Nemotron ASR** + **Parakeet TDT v3** (`nvidia/parakeet-tdt-0.6b-v2`) or **Faster-Whisper large-v3-turbo** | Parakeet TDT v3: `0.6B` (`~2.3 GB` disk / `~1.5 GB` RAM); `turbo`: `809M` (`~1.6 GB` disk) ([Weave](https://xlaude2040.com/weave/), [kami-subs](https://github.com/MohammdKopa/kami-subs)) | **Off-the-Shelf** | **In:** Streaming `16kHz` audio frames.<br>**Out:** `0.62s` median provisional word emission + endpoint verification landing invisibly in the `-1.75s` read-ahead zone ([Weave](https://xlaude2040.com/weave/)). |
| **Stream 1c: Browser Edge WebGPU ASR** | **Moonshine Base** (`onnx-community/moonshine-base-ONNX`) or **Whisper Base** (`onnx-community/whisper-base`) | Moonshine Base: `~61M` (`~65 MB` `q8`); Whisper Base: `74M` (`~145–210 MB`) ([Moonshine ASR](https://webai.show/models/moonshine-asr/), [local-voice-chat](https://github.com/Hari31416/local-voice-chat)) | **Off-the-Shelf** | **In:** Variable-length unpadded `16kHz` utterance.<br>**Out:** Client-side WebGPU transcription scaling linearly with utterance duration. |
| **Stream 2a: Primary Streaming Diarizer** | **NVIDIA Streaming Sortformer 4spk v2.1** (`nvidia/diar_streaming_sortformer_4spk-v2.1`) + **ERes2Net** (`iic/speech_eres2net_sv_en_voxceleb_16k`) / **CAM++** (`iic/speech_campplus_sv_ko_16k`) | `17-layer` NEST Fast-Conformer + `18-layer` Transformer (`hidden_size=192`); standalone VRAM undisclosed in model card ([Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)) | **Off-the-Shelf** | **In:** Streaming audio with `80 ms` frame step (`CHUNK_SIZE=6`, `RIGHT_CONTEXT=7`, `FIFO_SIZE=188`, `spkcache_size=188`).<br>**Out:** 4 sigmoid speaker probabilities per `80 ms` frame at `1.04s` latency (`5.09%` DER on 2-spk CH109) + `4.48s` median turn verification ([Weave](https://xlaude2040.com/weave/)). |
| **Stream 2b: Sub-Second / Low-RAM Diarizer Fallback** | **Pyannote Streaming** or **@Lokesh_Lal Custom Clustering Diarizer** | Custom fallback: **512 MB RAM** (down from `4 GB` standard diarization RAM) ([CwiRenderer](https://stardance.hackclub.com/projects/63123), [Pyannote](https://www.pyannote.ai/blog/how-we-built-streaming-diarization)) | **Off-the-Shelf** | **In:** Fixed `100 ms` chunks (`6,400 bytes` of `16kHz` `float32` PCM).<br>**Out:** Speaker cluster ID within a `300 ms` all-inclusive latency budget. |
| **Stream 3: Token-Level Sentence Stress Head** | **Custom WhiStress Layer-9 Head** (`slprl/WhiStress` architecture adapted to frozen CrisperWhisper 2.0 / Whisper-turbo Layer 9) | **`~7M` trainable parameters** (`~14 MB` FP16) attached to frozen Whisper encoder/decoder Layer 9 ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)) | **TRAIN / FINE-TUNE** | **In:** Whisper Encoder Layer-9 `[B, T_enc, D]` + Decoder Layer-9 `[B, T_dec, D]`.<br>**Out:** Alignment-free token-level prominence score $P \in [0.0, 1.0]$ (`0.909 F1` on TinyStress-15K, `0.961 F1` on Aix-MARSEC). |
| **Stream 4a: Audio Events & SFX Detector** | **SenseVoice-Small** (`iic/SenseVoiceSmall`) / **SenseVoice-Large** (`iic/SenseVoiceLarge` for VOD boundary timestamps) | Parameter count undisclosed in doc; Small runs **`~15×` faster than Whisper-large-v3** ([FunAudioLLM](https://fun-audio-llm.github.io/), [awesome-voice-agents](https://github.com/yzfly/awesome-voice-agents)) | **Off-the-Shelf** | **In:** `16kHz` audio segment.<br>**Out:** `<|BGM|>`, `<|Applause|>`, `[laughter]`, `[crying]`, `[coughing]`, `[sneezing]`, `[breathing]` + emotion tags (`<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`). |
| **Stream 4b: Vocal Affect & Texture Head** | **Multi-Task Head** on frozen **emotion2vec_plus_base** (`~90M`) / **emotion2vec_plus_large** (`~300M`, or `mlx-community/emotion2vec-plus-large-mlx`) / **emotion2vec-web-distill** (`thomashallock/emotion2vec-web-distill`, `2,231,986` params) | Frozen `~90M` / `~300M` / `2.23M` backbone + **`~0.8M` trainable linear/attentive head** ([emotion2vec+ Large](https://huggingface.co/emotion2vec/emotion2vec_plus_large), [web-distill](https://huggingface.co/thomashallock/emotion2vec-web-distill)) | **TRAIN / FINE-TUNE** | **In:** `50 Hz` (`20 ms` stride) frame embeddings `[B, T_50hz, 768/1024]`.<br>**Out:** Continuous Arousal/Valence/Dominance + 7-class human non-speech vocalization boundaries (`[sighs]`, `[gasps]`, `[screaming]`, `[yawning]`). |

---

## 3. ASR & Timestamp Alignment Benchmark Justification

Why **CrisperWhisper 2.0** (`nyrahealth/CrisperWhisper`) is selected over standard Whisper or two-stage forced aligners ([CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), [Nyra Labs Attention-to-Aligner](https://nyra-labs.com/research/attention-to-aligner)):

| ASR / Alignment System | TIMIT Read Speech Mean Absolute Boundary Error | Conversational Speech Mean Error | Architecture & Limitation |
| :--- | :--- | :--- | :--- |
| **CrisperWhisper 2.0** *(Selected VOD/Buffered)* | **29.6 ms** | **41.0 ms** | Applies Viterbi decoding directly to **supervised cross-attention alignment heads** inside a single CTranslate2 pass (+ `turbo` speculative decoding). Preserves verbatim `[um]`/`[uh]` fillers and breaks repetition loops. |
| **xAI Grok STT** | 37.1 ms | — | Proprietary cloud API (cannot run offline locally). |
| **CTC-seg** | 49.3 ms | — | Requires separate CTC segmentation pass. |
| **ElevenLabs Scribe v2** | 51.3 ms | — | Proprietary cloud API. |
| **NeMo-FA (Forced Aligner)** | 60.0 ms | — | Two-stage forced alignment overhead. |
| **Deepgram Nova-3** | 63.3 ms | — | Proprietary cloud API. |
| **WhisperX (`wav2vec2` aligner)** | 64.8 ms | — | Requires loading a second `wav2vec2` phoneme model per language; `64.8 ms` error exceeds 2 video frames at 30 fps. |
| **Cartesia Ink-Whisper** | 69.4 ms | — | Coarser timestamp precision. |
| **NVIDIA Canary** | 85.5 ms | — | Coarse subword timestamp boundaries (`>2.5` frames drift at 30 fps). |

---

## 4. Why WhiStress Taps Layer 9 of the Whisper Backbone

Random-forest regression probing of Whisper's encoder/decoder layers on **CREMA-D** ([WhiStress, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [arXiv:2505.19103v1](https://arxiv.org/html/2505.19103v1)) proves that deeper layers progressively discard acoustic $F_0$ pitch and RMS energy in favor of purely lexical/semantic tokens:

| Whisper Layer Tap Point | TinyStress-15K Stress Detection F1 | Aix-MARSEC Natural Speech Stress F1 | Acoustic vs. Semantic Representation Balance |
| :--- | :--- | :--- | :--- |
| **Layer 3** | `0.693` | — | Raw acoustic features present, but weak lexical token alignment. |
| **Layer 6** | `0.845` | — | Moderate acoustic + lexical fusion. |
| **Layer 9 (Optimal Tap Point)** | **`0.909`** | **`0.961`** | **Peak synergy of acoustic prosody ($F_0$, RMS energy) and contextual lexical semantics.** |
| **Layer 12 (Final Layer)** | `0.884` | — | Acoustic pitch and loudness representations degrade as final layer optimizes pure word identity. |

*Proportional Layer Mapping Across Whisper Backbones:*
- On **Whisper-small** (`12` encoder / `12` decoder layers): Tap **Encoder Layer 9 & Decoder Layer 9** (0-indexed `layer_idx = 8`).
- On **CrisperWhisper 2.0 Large / Whisper-large-v3** (`32` encoder / `32` decoder layers): The proportional prosodic peak at $9/12 = 75\%$ depth corresponds to **Encoder Layer 24** (`layer_idx = 23`) paired with **Decoder Layer 24** (or **Decoder Layer 3** on `large-v3-turbo`'s 4-layer decoder). Our `WhiStressLayer9Head` configurable `tap_layer_encoder` and `tap_layer_decoder` parameters handle both `small` (`9, 9`), `turbo` (`24, 3`), and `large` (`24, 24`) seamlessly.
