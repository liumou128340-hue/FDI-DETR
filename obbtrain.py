from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    model = YOLO(r"ultralytics/cfg/models/fdi-detr/fdidetr-s-obb.yaml")
    # model = YOLO(r"ultralytics/cfg/models/fdi-detr/fdidetr-m-obb.yaml")

    results = model.train(
        data='../CODrone/CODrone.yaml',
        imgsz=640, amp=False,deterministic=False,
        batch=8,workers=8,

        val=True, epochs=300, patience=20,
        task="obb",project="CODrone", name="AS640",

        optimizer="AdamW", lr0=0.001, cos_lr=True, lrf=0.1,
        close_mosaic=50, weight_decay=0.0005, warmup_epochs=5,warmup_bias_lr=0.0001,

    )
