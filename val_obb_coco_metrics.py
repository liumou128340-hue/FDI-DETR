from pathlib import Path
from collections import defaultdict
import json
import os

import yaml
from PIL import Image
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MODEL = r"CODrone/fdidetr_s_obb_1024/weights/best.pt"
DATA = r"/userA02/lht/CODrone/CODrone.yaml"
SPLIT = "val"
IMGSZ = 1024
BATCH = 8
WORKERS = 8
DEVICE = 0
PROJECT = "val/CODrone"
NAME = "fdidetr_s_obb_1024"
MAX_DET = 100

IOU_THRESHOLDS = [round(x / 100, 2) for x in range(50, 100, 5)]
AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}


def polygon_area(poly):
    pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _signed_area(pts):
    return sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    ) * 0.5


def _inside(p, a, b, orientation):
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return cross >= -1e-9 if orientation >= 0 else cross <= 1e-9


def _intersection(s, e, a, b):
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return e
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def _clip_polygon(subject, clip):
    if len(subject) < 3 or len(clip) < 3:
        return []
    output = subject
    orientation = 1 if _signed_area(clip) >= 0 else -1
    for i, a in enumerate(clip):
        b = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, a, b, orientation):
                if not _inside(s, a, b, orientation):
                    output.append(_intersection(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b, orientation):
                output.append(_intersection(s, e, a, b))
            s = e
    return output


def poly_iou(poly1, poly2):
    p1 = [(float(poly1[i]), float(poly1[i + 1])) for i in range(0, len(poly1), 2)]
    p2 = [(float(poly2[i]), float(poly2[i + 1])) for i in range(0, len(poly2), 2)]
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    inter_poly = _clip_polygon(p1, p2)
    if len(inter_poly) < 3:
        return 0.0
    inter = abs(_signed_area(inter_poly))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def resolve_dataset(data_yaml):
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", ""))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return data, root


def image_paths(root, split_value):
    paths = split_value if isinstance(split_value, list) else [split_value]
    out = []
    for item in paths:
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        if p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                q = Path(line.strip())
                if q:
                    out.append(q if q.is_absolute() else root / q)
        elif p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                out.extend(p.rglob(ext))
        else:
            raise FileNotFoundError(f"split path not found: {p}")
    return sorted(out)


def label_path_for_image(img_path):
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name


def image_id(path):
    return int(path.stem) if path.stem.isnumeric() else path.stem


def load_ground_truth(data_yaml, split):
    data, root = resolve_dataset(data_yaml)
    names = data["names"]
    nc = len(names)
    gts = defaultdict(list)
    image_ids = []
    for img_path in image_paths(root, data[split]):
        im_id = image_id(img_path)
        image_ids.append(im_id)
        with Image.open(img_path) as im:
            width, height = im.size
        label_path = label_path_for_image(img_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 9:
                continue
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
            poly = []
            for i, value in enumerate(coords):
                poly.append(value * (width if i % 2 == 0 else height))
            gts[(im_id, cls + 1)].append({"poly": poly, "area": polygon_area(poly), "matched": {}})
    return gts, image_ids, nc


def load_predictions(pred_json, image_ids):
    valid_ids = set(image_ids)
    detections = []
    for item in json.loads(Path(pred_json).read_text(encoding="utf-8")):
        im_id = item["image_id"]
        if im_id not in valid_ids:
            continue
        if "poly" not in item:
            continue
        detections.append(
            {
                "image_id": im_id,
                "category_id": int(item["category_id"]),
                "score": float(item["score"]),
                "poly": [float(x) for x in item["poly"]],
                "area": polygon_area(item["poly"]),
            }
        )
    return detections


def compute_ap(recalls, precisions):
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_area(gts, detections, nc, area_range, iou_thr):
    min_area, max_area = area_range
    gt_eval = {}
    ignored = {}
    total_gt = 0
    for key, items in gts.items():
        valid = []
        ignore = []
        for gt in items:
            gt_copy = {**gt, "matched": False}
            if min_area <= gt["area"] < max_area:
                valid.append(gt_copy)
            else:
                ignore.append(gt_copy)
        if valid:
            total_gt += len(valid)
            gt_eval[key] = valid
        if ignore:
            ignored[key] = ignore

    if total_gt == 0:
        return None

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    tp, fp = [], []

    per_image_count = defaultdict(int)
    for det in detections:
        image_key = (det["image_id"], det["category_id"])
        per_image_count[image_key] += 1
        if per_image_count[image_key] > MAX_DET:
            continue

        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_eval.get(image_key, [])):
            if gt["matched"]:
                continue
            iou = poly_iou(det["poly"], gt["poly"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thr and best_idx >= 0:
            gt_eval[image_key][best_idx]["matched"] = True
            tp.append(1)
            fp.append(0)
            continue

        ignore_det = False
        for gt in ignored.get(image_key, []):
            if poly_iou(det["poly"], gt["poly"]) >= iou_thr:
                ignore_det = True
                break
        if ignore_det:
            continue

        tp.append(0)
        fp.append(1)

    if not tp:
        return 0.0
    tp_cum, fp_cum = [], []
    running_tp = running_fp = 0
    for t, f in zip(tp, fp):
        running_tp += t
        running_fp += f
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)
    recalls = [x / total_gt for x in tp_cum]
    precisions = [tp_cum[i] / max(tp_cum[i] + fp_cum[i], 1) for i in range(len(tp_cum))]
    return compute_ap(recalls, precisions)


def evaluate_obb_coco_style(gt_data, detections, nc):
    gts, _, _ = gt_data
    results = {}
    for area_name, area_range in AREA_RANGES.items():
        aps = []
        ap50 = ap75 = None
        for thr in IOU_THRESHOLDS:
            ap = evaluate_area(gts, detections, nc, area_range, thr)
            if ap is not None:
                aps.append(ap)
            if thr == 0.50:
                ap50 = ap
            if thr == 0.75:
                ap75 = ap
        results[area_name] = sum(aps) / len(aps) if aps else -1.0
        if area_name == "all":
            results["AP50"] = ap50 if ap50 is not None else -1.0
            results["AP75"] = ap75 if ap75 is not None else -1.0
    return {
        "AP": results["all"],
        "AP50": results["AP50"],
        "AP75": results["AP75"],
        "APS": results["small"],
        "APM": results["medium"],
        "APL": results["large"],
    }


def main():
    model = YOLO(MODEL)
    save_dir = Path(PROJECT) / NAME

    model.val(
        task="obb",
        data=DATA,
        split=SPLIT,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        half=False,
        save_json=True,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        device=DEVICE,
    )

    pred_json = save_dir / "predictions.json"
    gt_data = load_ground_truth(DATA, SPLIT)
    detections = load_predictions(pred_json, gt_data[1])
    metrics = evaluate_obb_coco_style(gt_data, detections, gt_data[2])

    print("\nOBB COCO-style metrics using polygon IoU:")
    for key in ["AP", "AP50", "AP75", "APS", "APM", "APL"]:
        value = metrics[key]
        print(f"{key}: {value:.4f}" if value >= 0 else f"{key}: nan")


if __name__ == "__main__":
    main()
