from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
# benchmark(model="yolo11n.pt", data="coco8.yaml",
#           imgsz=640, half=False, device=0)

# Benchmark specific export format
# benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")


benchmark(model="yolo11n.pt", data="coco.yaml",
          imgsz=640, format="tflite", half=False, int8=True, device="mps")


# benchmark(model="/Users/tosaki/dev/YOLO-Infer-pt/weights_new/ultralytics_converted.pt", data="coco.yaml",
#           imgsz=640, format="openvino", half=False)
