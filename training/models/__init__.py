"""Neural network heads and adapters for CWI platform custom training.

Includes:
- WhiStressLayer9Head: 7M-parameter cross-attention decoder block attached to Whisper Layer 9.
- Emotion2VecMultiTaskHead: Attentive temporal pooling multi-task head on 50Hz emotion2vec+ frames.
"""

from training.models.whistress_layer9_head import WhiStressLayer9Head
from training.models.emotion2vec_multitask_head import Emotion2VecMultiTaskHead

__all__ = [
    "WhiStressLayer9Head",
    "Emotion2VecMultiTaskHead",
]
