from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")
print(model.model.model)

# Quant model.model.model