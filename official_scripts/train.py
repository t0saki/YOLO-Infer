from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.pt")
# # build from YAML and transfer weights
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")

model = YOLO("yolo11n_int8_openvino_model")

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
