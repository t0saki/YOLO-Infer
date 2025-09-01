from tflite_support import TFLiteModelInfo

# --- 请修改这里 ---
# 将此路径替换成你【原版正常的】tflite 模型的实际路径
MODEL_PATH = "yolov10n_int8.tflite"

# 定义输出的标签文件名
OUTPUT_LABEL_FILE = "labels.txt"
# --------------------

print(f"正在从 '{MODEL_PATH}' 模型中提取标签...")

try:
    # 加载模型信息
    info = TFLiteModelInfo.from_file(MODEL_PATH)

    # 从元数据中获取关联的标签文件内容
    # 标签文件的类型通常是 'TENSOR_AXIS_LABELS'
    label_file_content = info.get_associated_files("TENSOR_AXIS_LABELS")

    if label_file_content:
        # 将提取到的标签内容写入到输出文件中
        with open(OUTPUT_LABEL_FILE, "w", encoding="utf-8") as f:
            f.write(label_file_content)
        print(f"✅ 成功! 标签已提取并保存到 '{OUTPUT_LABEL_FILE}' 文件中。")
    else:
        print("❌ 失败: 在模型的元数据中没有找到标签文件。")

except Exception as e:
    print(f"发生错误: {e}")
    print("请检查:")
    print("1. `tflite-support` 库是否已正确安装。")
    print(f"2. 模型文件路径 '{MODEL_PATH}' 是否正确。")