"""Automated Caption with Intention (CWI) Platform - Training & Distillation Module.

This package contains custom model training scripts, datasets, and ONNX hybrid
distillation exporters for:
1. WhiStress Layer-9 Prominence Head (7M-param cross-attention decoder block attached
   to frozen Whisper Layer 9, detecting token-level sentence stress).
2. Emotion2Vec+ Multi-Task Head (~0.8M-param attentive pooling & continuous
   Valence/Arousal/Dominance + 7-class human non-speech vocalization classifier).
"""

from training.models.whistress_layer9_head import WhiStressLayer9Head
from training.models.emotion2vec_multitask_head import Emotion2VecMultiTaskHead

__all__ = [
    "WhiStressLayer9Head",
    "Emotion2VecMultiTaskHead",
]
