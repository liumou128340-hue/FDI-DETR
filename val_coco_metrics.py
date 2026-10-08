from pathlib import Path
import json
import os

import yaml
from PIL import Image
from ultralytics import RTDETR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MODEL = r"runs/detect/visdrone/stage/A-11/weights/best.pt"
DATA = r"ultralytics/cfg/datasets/uavaste.yaml"
SPLIT = "val"
IMGSZ = 640
BATCH = 8
WORKERS = 0
DEVICE = 3
PROJECT = "val"
NAME = "A"


def _resolve_dataset_path(data_yaml):
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", ""))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return data, root


def _image_paths(root, split_value):
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


def _label_path_for_image(img_path):
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name


def _image_id(path):
    return int(path.stem) if path.stem.isnumeric() else path.stem


def yolo_to_coco_json(data_yaml, split, out_json):
    data, root = _resolve_dataset_path(data_yaml)
    names = data["names"]
    if isinstance(names, list):
        categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]
    else:
        categories = [{"id": int(i) + 1, "name": name} for i, name in names.items()]

    images, annotations = [], []
    ann_id = 1
    for img_path in _image_paths(root, data[split]):
        with Image.open(img_path) as im:
            width, height = im.size
        image_id = _image_id(img_path)
        images.append({"id": image_id, "file_name": img_path.name, "width": width, "height": height})

        label_path = _label_path_for_image(img_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            x = (xc - bw / 2) * width
            y = (yc - bh / 2) * height
            w = bw * width
            h = bh * height
            x = max(0.0, x)
            y = max(0.0, y)
            w = max(0.0, min(w, width - x))
            h = max(0.0, min(h, height - y))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                    "area": round(w * h, 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {},
        "licenses": [],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return out_json


def coco_eval(gt_json, pred_json):
    try:
        from faster_coco_eval import COCO, COCOeval_faster

        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(pred_json))
        evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        s = evaluator.stats_as_dict
        return {
            "AP": s["AP_all"],
            "AP50": s["AP_50"],
            "AP75": s["AP_75"],
            "APS": s["AP_small"],
            "APM": s["AP_medium"],
            "APL": s["AP_large"],
        }
    except Exception:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(pred_json))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {
            "AP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]),
            "APS": float(evaluator.stats[3]),
            "APM": float(evaluator.stats[4]),
            "APL": float(evaluator.stats[5]),
        }


def main():
    model = RTDETR(MODEL)
    save_dir = Path(PROJECT) / NAME

    model.val(
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
    gt_json = save_dir / f"instances_{SPLIT}.json"
    yolo_to_coco_json(DATA, SPLIT, gt_json)
    metrics = coco_eval(gt_json, pred_json)

    print("\nCOCO metrics:")
    for k in ["AP", "AP50", "AP75", "APS", "APM", "APL"]:
        print(f"{k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
