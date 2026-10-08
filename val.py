from ultralytics import RTDETR


def main():
    model = RTDETR(r"runs/detect/uavaste/A-11/weights/best.pt")

    metrics = model.val(
        # data=r"ultralytics/cfg/datasets/Visdrone_Yolo.yaml",
        # data='ultralytics/cfg/datasets/UAVDT.yaml',
        # data='ultralytics/cfg/datasets/coco.yaml',
        data='ultralytics/cfg/datasets/uavaste.yaml',
        split="val",
        # split="test",
        imgsz=640,
        batch=8,
        workers=0,
        half=False,
        save_json=True,
        project="val/uavaste",
        name="A-11",
        device=3,
    )

    # print(metrics)


if __name__ == "__main__":
    main()
