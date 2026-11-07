"""WhiStress Layer-9 Prominence Head Module.

Adapted from Interspeech 2025 WhiStress (Yosha et al., arXiv:2505.19103v1).
Attaches a 7M-parameter Cross-Attention Decoder Block and 2-layer FCNN
binary token prominence classifier to Layer 9 of frozen Whisper backbones
(e.g., openai/whisper-small.en, nyralabs/CrisperWhisper2.0_large, or
openai/whisper-large-v3-turbo).
"""

from typing import Optional
import torch
import torch.nn as nn


class WhiStressLayer9Head(nn.Module):
    """7M-parameter Stress Detection Head (Interspeech 2025 slprl/WhiStress architecture):
    - 1 additional Transformer Decoder Block with self-attention + cross-attention
      attending to Encoder Layer-9 prosodic hidden states
    - 2-layer FCNN binary token classifier outputting stress logits per decoder token.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

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
        decoder_hidden_l9: Optional[torch.Tensor] = None,
        encoder_hidden_l9: Optional[torch.Tensor] = None,
        decoder_key_padding_mask: Optional[torch.Tensor] = None,
        decoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_layer9: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the WhiStress Layer-9 head.

        Args:
            decoder_hidden_l9: [B, T_dec, D] decoder hidden states (Layer 9).
            encoder_hidden_l9: [B, T_enc, D] encoder hidden states (Layer 9).
            decoder_key_padding_mask: Optional [B, T_dec] mask for self-attention.
            decoder_hidden_states: Alias for decoder_hidden_l9.
            encoder_hidden_states_layer9: Alias for encoder_hidden_l9.
            attention_mask: Alias for decoder_key_padding_mask.

        Returns:
            torch.Tensor of shape [B, T_dec, 1] containing stress logits.
        """
        if decoder_hidden_l9 is None:
            decoder_hidden_l9 = decoder_hidden_states
        if encoder_hidden_l9 is None:
            encoder_hidden_l9 = encoder_hidden_states_layer9
        if decoder_key_padding_mask is None:
            decoder_key_padding_mask = attention_mask

        if decoder_hidden_l9 is None or encoder_hidden_l9 is None:
            raise ValueError("Both decoder_hidden_l9 and encoder_hidden_l9 tensors are required.")

        x = decoder_hidden_l9
        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=decoder_key_padding_mask)
        x = self.norm1(x + sa_out)

        ca_out, _ = self.cross_attn(query=x, key=encoder_hidden_l9, value=encoder_hidden_l9)
        x = self.norm2(x + ca_out)

        x = self.norm3(x + self.ffn(x))
        logits = self.classifier(x)  # [B, T_dec, 1]
        return logits
