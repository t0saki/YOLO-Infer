from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Export the model
model.export(format="saved_model", half=False, int8=True,
             device="mps", data="coco.yaml")
