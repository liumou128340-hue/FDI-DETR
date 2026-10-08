from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import RTDETR, YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Default run config. Edit these values if you want to run `python heatmap.py`
# without typing command-line arguments every time.
DEFAULT_MODEL = r"runs/detect/visdrone/stage/A-23P/weights/best.pt"
# DEFAULT_MODEL = r"runs/detect/visdrone/moduel/baseline/weights/best.pt"
DEFAULT_WEIGHTS = ""
DEFAULT_SOURCE = r"runs\visualization\forwards\ori3.jpg"
DEFAULT_SAVE_DIR = r"runs\visualization\forwards\heatmap"
DEFAULT_LAYER = "1,3,5,7,10,15,20,25,28,31"
# DEFAULT_LAYER = "1,3,7,9,12,17,21,24,27"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_TOPK = 1
DEFAULT_ALPHA = 0.45
DEFAULT_DEVICE = "0"
DEFAULT_HALF = False
DEFAULT_DRAW_BOXES = False


class MultiLayerGradCAM:
    """Minimal multi-layer Grad-CAM implementation for Ultralytics detection models."""

    def __init__(self, model: torch.nn.Module, layers: list[torch.nn.Module]):
        self.model = model
        self.layers = layers
        self.activations: dict[int, torch.Tensor] = {}
        self.gradients: dict[int, torch.Tensor] = {}
        self.handles = []
        for idx, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._save_activation(idx)))
            self.handles.append(layer.register_full_backward_hook(self._save_gradient(idx)))

    @staticmethod
    def _first_tensor(output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError(f"Target layer output is not a tensor: {type(output)}")
        return output

    def _save_activation(self, idx: int):
        def hook(module, inputs, output):
            self.activations[idx] = self._first_tensor(output)

        return hook

    def _save_gradient(self, idx: int):
        def hook(module, grad_input, grad_output):
            self.gradients[idx] = self._first_tensor(grad_output[0])

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, score: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        cams = []
        for idx in range(len(self.layers)):
            if idx not in self.activations or idx not in self.gradients:
                continue
            act = self.activations[idx]
            grad = self.gradients[idx]
            if act.ndim == 3:
                act = act.unsqueeze(0)
            if grad.ndim == 3:
                grad = grad.unsqueeze(0)
            if act.ndim != 4 or grad.ndim != 4:
                continue

            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().float().cpu().numpy()
            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cams.append(cam)

        if not cams:
            raise RuntimeError("Target layers did not produce usable CAMs. Try --layer 25,28,31 or another neck layer.")

        fused = np.mean(np.stack(cams, axis=0), axis=0)
        fused -= fused.min()
        fused /= fused.max() + 1e-8
        return fused


def guess_model_api(model_path: str):
    suffix = Path(model_path).suffix.lower()
    text = model_path.lower()
    if "rtdetr" in text or "detr" in text:
        return RTDETR
    if suffix in {".yaml", ".yml"} and "rtdetr" in text:
        return RTDETR
    return YOLO


def resolve_layer(model: torch.nn.Module, layer: str | int) -> torch.nn.Module:
    if isinstance(layer, int) or str(layer).lstrip("-").isdigit():
        return model.model[int(layer)]

    current = model
    for part in str(layer).split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def resolve_layers(model: torch.nn.Module, layers: str) -> list[torch.nn.Module]:
    return [resolve_layer(model, part.strip()) for part in str(layers).split(",") if part.strip()]


def letterbox_bgr(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def image_to_tensor(image_bgr: np.ndarray, device: torch.device, half: bool) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).to(device).permute(2, 0, 1).contiguous().float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor.half() if half else tensor


def iter_images(source: str | Path) -> Iterable[Path]:
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path
    else:
        yield source


def pick_score(raw_output, conf: float, topk: int) -> torch.Tensor:
    """Return a scalar detection score for backpropagation."""
    out = raw_output

    # RT-DETR eval returns (postprocessed_predictions, raw_decoder_outputs).
    # Use raw decoder class scores for Grad-CAM because postprocess/top-k output
    # may be detached or otherwise unsuitable for backprop in some Ultralytics builds.
    if isinstance(out, (list, tuple)):
        if len(out) >= 2 and isinstance(out[1], (list, tuple)) and len(out[1]) >= 2 and torch.is_tensor(out[1][1]):
            scores = out[1][1]
            if scores.ndim == 4:  # [decoder_layers, batch, queries, classes]
                scores = scores[-1, 0].sigmoid().flatten()
            elif scores.ndim == 3:  # [batch, queries, classes]
                scores = scores[0].sigmoid().flatten()
            else:
                scores = scores.sigmoid().flatten()
            keep = scores > conf
            scores = scores[keep] if keep.any() else scores
            return scores.topk(min(topk, scores.numel())).values.sum()

        # Some training/raw paths return (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta).
        if len(out) >= 2 and torch.is_tensor(out[1]):
            scores = out[1]
            if scores.ndim == 4:
                scores = scores[-1, 0].sigmoid().flatten()
            elif scores.ndim == 3:
                scores = scores[0].sigmoid().flatten()
            else:
                scores = scores.sigmoid().flatten()
            keep = scores > conf
            scores = scores[keep] if keep.any() else scores
            return scores.topk(min(topk, scores.numel())).values.sum()

        out = out[0]

    # RT-DETR eval output is usually [B, N, 6] with score at index 4.
    if torch.is_tensor(out) and out.ndim == 3 and out.shape[-1] >= 6:
        scores = out[0, :, 4]
        keep = scores > conf
        scores = scores[keep] if keep.any() else scores
        return scores.topk(min(topk, scores.numel())).values.sum()

    # YOLO Detect eval output is often [B, 4 + nc, N].
    if torch.is_tensor(out) and out.ndim == 3:
        pred = out[0]
        cls_scores = pred[4:, :] if pred.shape[0] > pred.shape[1] else pred[:, 4:]
        if cls_scores.numel() == 0:
            return pred.max()
        scores = cls_scores.max(dim=0).values if cls_scores.shape[0] < cls_scores.shape[1] else cls_scores.max(dim=1).values
        keep = scores > conf
        scores = scores[keep] if keep.any() else scores
        return scores.topk(min(topk, scores.numel())).values.sum()

    if isinstance(out, dict):
        for key in ("pred_scores", "scores", "logits"):
            if key in out and torch.is_tensor(out[key]):
                scores = out[key].sigmoid().flatten()
                return scores.topk(min(topk, scores.numel())).values.sum()

    if torch.is_tensor(out):
        return out.flatten().topk(min(topk, out.numel())).values.sum()

    raise RuntimeError(f"Unsupported model output type: {type(raw_output)}")


def overlay_heatmap(original_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap, alpha, 0)


def draw_boxes(model_wrapper, image_path: Path, image_bgr: np.ndarray, imgsz: int, conf: float, device: str) -> np.ndarray:
    result = model_wrapper.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    plotted = result.plot(labels=False, conf=False)
    return plotted


def torch_device(device_arg: str) -> torch.device:
    value = str(device_arg).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value if torch.cuda.is_available() else "cpu")
    if value.lstrip("-").isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{value}")
    return torch.device("cpu")


def run(args):
    device = torch_device(args.device)
    api = guess_model_api(args.model)
    wrapper = api(args.model)
    if args.weights:
        wrapper.load(args.weights)

    net = wrapper.model.to(device).eval()
    for param in net.parameters():
        param.requires_grad_(True)
    if args.half and device.type != "cpu":
        net.half()

    layers = resolve_layers(net, args.layer)
    cam_runner = MultiLayerGradCAM(net, layers)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        for image_path in iter_images(args.source):
            original = cv2.imread(str(image_path))
            if original is None:
                print(f"Skip unreadable image: {image_path}")
                continue

            input_img, scale, (pad_x, pad_y) = letterbox_bgr(original, args.imgsz)
            tensor = image_to_tensor(input_img, device, args.half)

            output = net(tensor)
            score = pick_score(output, args.conf, args.topk)
            cam_square = cam_runner(score, (args.imgsz, args.imgsz))

            h, w = original.shape[:2]
            crop_w = int(round(w * scale))
            crop_h = int(round(h * scale))
            cam_crop = cam_square[pad_y : pad_y + crop_h, pad_x : pad_x + crop_w]
            cam = cv2.resize(cam_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            cam -= cam.min()
            cam /= cam.max() + 1e-8

            overlay = overlay_heatmap(original, cam, args.alpha)
            if args.draw_boxes:
                boxed = draw_boxes(wrapper, image_path, original, args.imgsz, args.conf, args.device)
                overlay = cv2.addWeighted(overlay, 0.65, boxed, 0.35, 0)

            out_path = save_dir / f"{image_path.stem}_heatmap.jpg"
            cv2.imwrite(str(out_path), overlay)
            print(f"Saved: {out_path}")
    finally:
        cam_runner.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM heatmap for Ultralytics YOLO/RT-DETR models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .pt/.yaml model.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Optional weights path when --model is a yaml.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image file or image directory.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory to save heatmaps.")
    parser.add_argument("--layer", default=DEFAULT_LAYER, help="Target layer index/module path, comma-separated for fusion. Example: 25,28,31")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Use top-k detection scores as Grad-CAM target.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Heatmap overlay alpha.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA id like 0, or cpu.")
    parser.add_argument("--half", action="store_true", default=DEFAULT_HALF, help="Use FP16 on CUDA.")
    parser.add_argument("--draw-boxes", action="store_true", default=DEFAULT_DRAW_BOXES, help="Blend predicted boxes into the heatmap image.")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run(parse_args())
