from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")

# 导出 FP32/FP16
model.export(format="tflite", imgsz=640, batch=1, dynamic=False, nms=False, simplify=True)  # FP32

os.rename("yolo11n_saved_model", "yolo11n_fp32")

model.export(format="tflite", imgsz=640, batch=1, dynamic=False, half=True, nms=False, simplify=True)  # FP16（权重半精度）

os.rename("yolo11n_saved_model", "yolo11n_fp16")

# 导出 INT8（需标注校准数据 data=...）
model.export(format="tflite", imgsz=640, batch=1, dynamic=False, int8=True, data="coco8.yaml", nms=False, simplify=True)

os.rename("yolo11n_saved_model", "yolo11n_int8")
