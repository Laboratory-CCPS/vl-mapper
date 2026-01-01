#!/usr/bin/env python3
"""
Export YOLO-World to ONNX with RAW predictions (no post-processing).
Post-processing (NMS, confidence filtering) will be done in C++.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics.models import YOLOWorld
import os



def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO-World to ONNX (raw predictions)")
    p.add_argument("--weights", type=str, required=True, help="Path to YOLO-World .pt weights")
    p.add_argument("--out-dir", type=str, default="onnx_exports_raw", help="Output directory for ONNX files")
    p.add_argument("--imgsz", type=int, nargs=2, default=[640, 640], metavar=("H", "W"), help="Export image size H W")
    p.add_argument("--max-labels", type=int, default=10, help="Maximum number of labels supported at runtime")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return p.parse_args()


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class ClipTextEncoderWrapper(torch.nn.Module):
    """Expose forward(tokens) -> text_embeddings using Ultralytics' CLIP wrapper."""

    def __init__(self, clip_module: torch.nn.Module):
        super().__init__()
        self.clip = clip_module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.clip, "encode_text"):
            raise RuntimeError("Provided clip_module does not implement encode_text(tokens)")
        feats = self.clip.encode_text(input_ids)
        if not torch.is_tensor(feats):
            raise RuntimeError("clip.encode_text did not return a tensor")
        return feats


class YoloVisionWrapper(torch.nn.Module):
    """Wrap YOLO-World vision backbone - returns RAW predictions for C++ post-processing."""

    def __init__(self, yolo_vision: torch.nn.Module, max_labels: int = 10):
        super().__init__()
        self.yolo = yolo_vision
        self.max_labels = int(max_labels)

    def forward(self, image: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if text_embeddings.dim() != 2:
            raise RuntimeError(
                f"text_embeddings must be 2D [num_labels, embed_dim], got {tuple(text_embeddings.shape)}"
            )
        # Slice up to max_labels
        txt = text_embeddings[: self.max_labels]

        # Inject embeddings into YOLO-World vision model
        self.yolo.txt_feats = txt.unsqueeze(0)

        # Get raw predictions - do NOT post-process
        raw_preds = self.yolo(image)
        if isinstance(raw_preds, (list, tuple)):
            raw_preds = raw_preds[0]
        elif isinstance(raw_preds, dict):
            raw_preds = next(iter(raw_preds.values()))
        
        return raw_preds  # Return raw [1, num_classes+4, num_anchors]


def slim_onnx_model(onnx_path: str, slimmed_path: str) -> None:
    import onnx
    import onnxslim

    model = onnx.load(onnx_path)
    slim_model = onnxslim.slim(model)
    if not slim_model:
        print(f"⚠️  Warning: ONNX simplification failed for {onnx_path}")
        return
    onnx.save(slim_model, slimmed_path)
    print(f"✔ Simplified ONNX model saved to {slimmed_path}")


def main() -> None:
    args = parse_args()
    # Allow users to pass ~ and environment variables in paths (e.g. "~/models/foo.pt" or "$HOME/models")
    def _expand(p: str) -> str:
        return os.path.expanduser(os.path.expandvars(p)) if isinstance(p, str) else p

    args.weights = _expand(args.weights)
    args.out_dir = _expand(args.out_dir)
    device = get_device()

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    # Load model
    model = YOLOWorld(args.weights)

    # Prepare dummy labels
    dummy_labels = [f"object_{i+1}" for i in range(min(args.max_labels, 10))]
    model.set_classes(dummy_labels)

    # Access internals
    yolo_vision = model.model
    clip_module = model.model.clip_model

    yolo_vision.to(device).eval()
    clip_module.to(device).eval()

    # Determine embed_dim
    try:
        embed_dim = int(yolo_vision.txt_feats.shape[-1])
    except Exception:
        embed_dim = 512

    # Export CLIP text encoder
    text_encoder = ClipTextEncoderWrapper(clip_module).to(device).eval()
    seq_len = 77
    num_trace_labels = min(args.max_labels, 10)
    dummy_ids = torch.randint(0, 100, (num_trace_labels, seq_len), dtype=torch.long, device=device)

    clip_onnx = str(out_dir / "clip_text_encoder.onnx")
    torch.onnx.export(
        text_encoder,
        (dummy_ids,),
        clip_onnx,
        input_names=["input_ids"],
        output_names=["text_embeddings"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"✔ Exported CLIP text encoder -> {clip_onnx}")
    
    slim_onnx_model(clip_onnx, str(out_dir / "clip_text_encoder_slim.onnx"))

    # Export YOLO-World vision model (RAW predictions)
    H, W = args.imgsz
    B = 1

    dummy_img = torch.randn(B, 3, H, W, device=device)
    dummy_embeds = torch.randn(num_trace_labels, embed_dim, device=device)

    # Set export mode
    try:
        yolo_vision.export = True
    except Exception:
        pass

    vision_wrapper = YoloVisionWrapper(yolo_vision, max_labels=args.max_labels).to(device).eval()

    vision_onnx = str(out_dir / "yoloworld_vision.onnx")
    torch.onnx.export(
        vision_wrapper,
        (dummy_img, dummy_embeds),
        vision_onnx,
        input_names=["image", "text_embeddings"],
        output_names=["raw_predictions"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"✔ Exported YOLO-World vision model -> {vision_onnx}")

    slim_onnx_model(vision_onnx, str(out_dir / "yoloworld_vision_slim.onnx"))

    print("Done. Raw predictions will be post-processed in C++.")


if __name__ == "__main__":
    main()