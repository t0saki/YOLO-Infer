

# model = YOLO("yolo11n_int8_openvino_model")

# results = model.val(data="coco128.yaml", plots=True)
# print(results.confusion_matrix.to_df())
from ultralytics import YOLO

# model = YOLO(
#     "/Users/tosaki/dev/YOLO-Infer-pt/weights/v11_n_ultralytics_format.pt")

model = YOLO(
    "/Users/tosaki/dev/YOLO-Infer/yolo11n_saved_model/yolo11n_full_integer_quant.tflite")
results = model.val(data="coco128.yaml", plots=True, device="cpu")
