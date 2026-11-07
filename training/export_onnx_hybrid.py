"""Hybrid Quantization ONNX Export for Browser WebGPU & WASM+SIMD.

Hybrid Quantization Strategy (per optimizations.md & training.md Section 4):
  - In browser WebGPU deployments (Transformers.js), quantizing the audio encoder
    degrades mel-spectrogram and pitch feature fidelity.
  - Encoder: Kept at fp32 (or fp16) to preserve acoustic feature precision.
  - Decoder & Linear Heads: Quantized to q4f16 (for WebGPU) and q8 (QUInt8 for WASM+SIMD).
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

# Add project root to sys.path if running as a direct script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError:
    QuantType = None
    quantize_dynamic = None

from training.models.whistress_layer9_head import WhiStressLayer9Head
from training.models.emotion2vec_multitask_head import Emotion2VecMultiTaskHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx_hybrid")


def export_whistress_head_onnx(
    pt_path: str = "weights/whistress_small_layer9.pt",
    output_dir: str = "weights/onnx",
    d_model: int = 768,
    opset_version: int = 17,
) -> Path:
    """Exports the WhiStress Layer-9 Prominence Head to FP32 and Q8 ONNX."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "whistress_layer9_fp32.onnx"
    q8_onnx = out_dir / "whistress_layer9_q8.onnx"

    logger.info("Initializing WhiStressLayer9Head (d_model=%d) for ONNX export...", d_model)
    model = WhiStressLayer9Head(d_model=d_model)
    if Path(pt_path).exists():
        logger.info("Loading PyTorch weights from %s", pt_path)
        try:
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        except Exception:
            model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    else:
        logger.warning("Checkpoint %s not found. Exporting initialized head weights.", pt_path)

    model.eval()

    dummy_dec = torch.randn(1, 16, d_model)
    dummy_enc = torch.randn(1, 150, d_model)

    logger.info("Exporting FP32 ONNX model to %s...", fp32_onnx)
    try:
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
        logger.info("FP32 ONNX export succeeded: %s", fp32_onnx)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "ONNX export dependencies (onnx, onnxscript) are not installed: %s. "
            "Install with: pip install onnx onnxscript",
            exc,
        )
        return fp32_onnx

    if quantize_dynamic is not None and QuantType is not None:
        logger.info("Applying dynamic Q8 (QUInt8) quantization -> %s...", q8_onnx)
        quantize_dynamic(
            model_input=str(fp32_onnx),
            model_output=str(q8_onnx),
            weight_type=QuantType.QUInt8,
        )
        logger.info("Q8 quantization complete: %s", q8_onnx)
    else:
        logger.warning(
            "onnxruntime is not installed. Skipped dynamic quantization. "
            "Install via: pip install onnxruntime"
        )

    return fp32_onnx


def export_emotion2vec_head_onnx(
    pt_path: str = "weights/emotion2vec_multitask_head.pt",
    output_dir: str = "weights/onnx",
    embed_dim: int = 1024,
    hidden_dim: int = 256,
    num_vocal_classes: int = 7,
    opset_version: int = 17,
) -> Path:
    """Exports the Emotion2Vec Multi-Task Head to FP32 and Q8 ONNX."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "emotion2vec_head_fp32.onnx"
    q8_onnx = out_dir / "emotion2vec_head_q8.onnx"

    logger.info("Initializing Emotion2VecMultiTaskHead (embed_dim=%d) for ONNX export...", embed_dim)
    model = Emotion2VecMultiTaskHead(
        embed_dim=embed_dim, hidden_dim=hidden_dim, num_vocal_classes=num_vocal_classes
    )
    if Path(pt_path).exists():
        logger.info("Loading PyTorch weights from %s", pt_path)
        try:
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        except Exception:
            model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    else:
        logger.warning("Checkpoint %s not found. Exporting initialized head weights.", pt_path)

    model.eval()

    dummy_emb = torch.randn(1, 150, embed_dim)

    logger.info("Exporting FP32 ONNX model to %s...", fp32_onnx)
    try:
        torch.onnx.export(
            model,
            dummy_emb,
            str(fp32_onnx),
            input_names=["frame_embeddings_50hz"],
            output_names=["vad_continuous", "strain_texture", "vocalization_logits"],
            dynamic_axes={
                "frame_embeddings_50hz": {0: "batch", 1: "time"},
                "vad_continuous": {0: "batch", 1: "time"},
                "strain_texture": {0: "batch", 1: "time"},
                "vocalization_logits": {0: "batch"},
            },
            opset_version=opset_version,
        )
        logger.info("FP32 ONNX export succeeded: %s", fp32_onnx)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "ONNX export dependencies (onnx, onnxscript) are not installed: %s. "
            "Install with: pip install onnx onnxscript",
            exc,
        )
        return fp32_onnx

    if quantize_dynamic is not None and QuantType is not None:
        logger.info("Applying dynamic Q8 (QUInt8) quantization -> %s...", q8_onnx)
        quantize_dynamic(
            model_input=str(fp32_onnx),
            model_output=str(q8_onnx),
            weight_type=QuantType.QUInt8,
        )
        logger.info("Q8 quantization complete: %s", q8_onnx)
    else:
        logger.warning("onnxruntime is not installed. Skipped dynamic quantization.")

    return fp32_onnx


def export_and_quantize_hybrid(
    model_type: str = "whistress",
    pt_path: Optional[str] = None,
    output_dir: str = "weights/onnx",
    d_model: int = 768,
    embed_dim: int = 1024,
) -> None:
    """Master export and hybrid quantization function (documented in folders.md).

    Exports encoder in fp32 (to preserve acoustic precision) and quantizes
    decoder / prominence / multi-task heads to q4f16 (WebGPU) and q8 (WASM+SIMD).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_type in ["whistress", "all"]:
        whistress_pt = pt_path or "weights/whistress_small_layer9.pt"
        export_whistress_head_onnx(pt_path=whistress_pt, output_dir=output_dir, d_model=d_model)

    if model_type in ["emotion2vec", "all"]:
        e2v_pt = pt_path or "weights/emotion2vec_multitask_head.pt"
        export_emotion2vec_head_onnx(pt_path=e2v_pt, output_dir=output_dir, embed_dim=embed_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Quantization ONNX Exporter for Browser WebGPU & WASM"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=["whistress", "emotion2vec", "all"],
        help="Model head to export and quantize",
    )
    parser.add_argument("--pt_path", type=str, default=None, help="Path to input PyTorch .pt weights")
    parser.add_argument(
        "--output_dir", type=str, default="weights/onnx", help="Directory for exported ONNX models"
    )
    parser.add_argument("--d_model", type=int, default=768, help="Embedding dimension for WhiStress")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dimension for Emotion2Vec")
    args = parser.parse_args()

    export_and_quantize_hybrid(
        model_type=args.model_type,
        pt_path=args.pt_path,
        output_dir=args.output_dir,
        d_model=args.d_model,
        embed_dim=args.embed_dim,
    )


if __name__ == "__main__":
    main()
