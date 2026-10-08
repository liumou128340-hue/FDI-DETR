import os
from ultralytics import RTDETR
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == '__main__':

    # model = YOLO("yolov8s.yaml")
    model = RTDETR(r"ultralytics/cfg/models/fdi-detr/fdidetr-m.yaml")

    model.load("rtdetr-l.pt")

    results = model.train(
                            # data='ultralytics/cfg/datasets/Visdrone_Yolo.yaml',
                            # data='ultralytics/cfg/datasets/uavaste.yaml',
                            # data='../aitod/aitod.yaml',
                            data='../tinyperson/tinyperson.yaml',
                            imgsz=640,amp=False,single_cls=True,deterministic=False,
                            batch=8, workers=8,

                            project="tinyperson",name='AM640',
                            val=True,epochs=200,patience=0,

                            optimizer='AdamW', lr0=0.001, cos_lr=True,lrf=0.1,
                            close_mosaic=40,weight_decay=0.0005,warmup_epochs=5,warmup_bias_lr=0.0001,
                            # mosaic=0.0,mixup=0.0,copy_paste=0.0,erasing=0.0,auto_augment=None,
                            # scale=0.2,translate=0.05,fliplr=0.5,flipud=0.0,
                            # hsv_h=0.005,hsv_s=0.25,hsv_v=0.20,
                            # degrees=0.0,shear=0.0,perspective=0.0,
                          )
