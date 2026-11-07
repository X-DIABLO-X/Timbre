"""Emotion2Vec+ Multi-Task Downstream Head Module.

Lightweight multi-task head (~0.8M parameters) attached to frozen 50Hz frame
embeddings from emotion2vec_plus_large (or emotion2vec_plus_base):
  1. Frame-level continuous Arousal, Valence, Dominance (normalized [0, 1] on SAM scale)
  2. Frame-level vocal strain / timbre contrast driving Roboto Flex XOPQ & YOPQ axes in [-1, 1]
  3. Utterance-level 7-class human non-speech vocalization classifier via attentive temporal pooling
     (laughter, sighs, gasps, crying, coughing, screaming, yawning)
"""

from typing import Dict
import torch
import torch.nn as nn


class Emotion2VecMultiTaskHead(nn.Module):
    """Lightweight multi-task head (~0.8M params) attached to frozen 50Hz emotion2vec+ features:
    1. Frame-level continuous Arousal, Valence, Dominance (normalized [0, 1])
    2. Frame-level vocal strain / timbre contrast driving Roboto Flex XOPQ & YOPQ
    3. Utterance-level 7-class human non-speech vocalization classifier
       (laughter, sighs, gasps, crying, coughing, screaming, yawning)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        num_vocal_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_vocal_classes = num_vocal_classes

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Frame-level continuous [Arousal, Valence, Dominance] in [0, 1]
        self.vad_head = nn.Linear(hidden_dim, 3)

        # Frame-level [XOPQ_delta, YOPQ_delta] in [-1, 1]
        self.texture_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

        # Attentive temporal pooling for 7-class non-speech vocalization detection
        self.attn_weights = nn.Linear(hidden_dim, 1)
        self.vocalization_head = nn.Linear(hidden_dim, num_vocal_classes)

    def forward(self, frame_embeddings_50hz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the multi-task head.

        Args:
            frame_embeddings_50hz: [B, T_50hz, embed_dim] frame representations.

        Returns:
            Dictionary containing:
                "vad_continuous": [B, T_50hz, 3] continuous SAM Arousal/Valence/Dominance in [0, 1]
                "strain_texture": [B, T_50hz, 2] XOPQ/YOPQ deltas in [-1, 1]
                "vocalization_logits": [B, num_vocal_classes] logits for human vocalizations
        """
        # frame_embeddings_50hz: [B, T_50hz, embed_dim]
        h = self.proj(frame_embeddings_50hz)  # [B, T, hidden_dim]
        vad_continuous = torch.sigmoid(self.vad_head(h))  # [B, T, 3]
        strain_texture = self.texture_head(h)  # [B, T, 2]

        attn = torch.softmax(self.attn_weights(h), dim=1)  # [B, T, 1]
        pooled = torch.sum(h * attn, dim=1)  # [B, hidden_dim]
        vocalization_logits = self.vocalization_head(pooled)  # [B, num_vocal_classes]

        return {
            "vad_continuous": vad_continuous,
            "strain_texture": strain_texture,
            "vocalization_logits": vocalization_logits,
        }
