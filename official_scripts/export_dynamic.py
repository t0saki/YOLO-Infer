from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="openvino", dynamic=True, int8=True, data="coco.yaml")
