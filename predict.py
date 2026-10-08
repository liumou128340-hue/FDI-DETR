from ultralytics import RTDETR

def main():
    # model = RTDETR(r"runs\detect\visdrone\stage\A-23P\weights\best.pt")
    model = RTDETR("runs/detect/visdrone/moduel/baseline/weights/best.pt")
    results = model.predict(
        source=r"runs\visualization\forwards\ori2.jpg",  # 单张图片路径
        imgsz=640,
        device=0,          # 或 "cpu"
        half=False,
        save=True,         # 保存带框的可视化图
        project="../visualization",
        name="forwards",
        hide_labels=True,  # 可视化图上不画类别名
        hide_conf=True,  # 可视化图上不画置信度
    )


if __name__ == "__main__":
    main()