from ultralytics import YOLO

model = YOLO("yolo11n_int8_openvino_model")

results = model.val(data="coco128.yaml", plots=True)
print(results.confusion_matrix.to_df())
