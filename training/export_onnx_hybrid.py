"""
Exports and hybrid-quantizes (`fp32` encoder / `QUInt8` head) both custom heads
(`WhiStressLayer9Head` and `Emotion2VecMultiTaskHead`) with full dynamic sequence axes.
"""
from pathlib import Path
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from training.models.whistress_layer9_head import WhiStressLayer9Head
from training.models.emotion2vec_multitask_head import Emotion2VecMultiTaskHead


def export_whistress_head_onnx(
    pt_path: str = "weights/whistress_small_layer9.pt",
    output_dir: str = "weights/onnx",
    d_model: int = 768,
    opset_version: int = 17,
) -> Path:
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
        opset_version=opset_version,
    )

    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(q8_onnx),
        weight_type=QuantType.QUInt8,
    )
    return q8_onnx


def export_emotion2vec_head_onnx(
    pt_path: str = "weights/emotion2vec_multitask_head.pt",
    output_dir: str = "weights/onnx",
    embed_dim: int = 1024,
    hidden_dim: int = 256,
    num_vocal_classes: int = 7,
    opset_version: int = 17,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "emotion2vec_head_fp32.onnx"
    q8_onnx = out_dir / "emotion2vec_head_q8.onnx"

    model = Emotion2VecMultiTaskHead(
        embed_dim=embed_dim, hidden_dim=hidden_dim, num_vocal_classes=num_vocal_classes
    )
    if Path(pt_path).exists():
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_in = torch.randn(1, 50, embed_dim)

    torch.onnx.export(
        model,
        (dummy_in,),
        str(fp32_onnx),
        input_names=["frame_embeddings_50hz"],
        output_names=["vad_continuous", "strain_texture", "vocalization_logits"],
        dynamic_axes={
            "frame_embeddings_50hz": {0: "batch", 1: "frames_50hz"},
            "vad_continuous": {0: "batch", 1: "frames_50hz"},
            "strain_texture": {0: "batch", 1: "frames_50hz"},
            "vocalization_logits": {0: "batch"},
        },
        opset_version=opset_version,
    )

    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(q8_onnx),
        weight_type=QuantType.QUInt8,
    )
    return q8_onnx
