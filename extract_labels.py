from tflite_support import schema_py_generated as _schema_fb
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os

def extract_labels(model_path, export_path='labels.txt'):
    """
    从 TFLite 模型的元数据中提取标签并保存到文件。

    参数:
        model_path (str): 原始 .tflite 模型的路径。
        export_path (str): 导出的标签文件路径。
    """
    try:
        # 加载模型
        displayer = _metadata_fb.MetadataDisplayer.with_model_file(model_path)

        # 提取与第一个输出张量关联的标签文件
        # 大多数分类模型的标签都在第一个（也是唯一一个）输出张量上
        associated_files = displayer.get_output_tensor_metadata()[0].associatedFiles

        if not associated_files:
            print(f"错误：在模型 '{model_path}' 的元数据中找不到关联的标签文件。")
            return

        # 获取标签文件的内容
        # 通常标签文件就是第一个关联文件
        labels_content = displayer.get_associated_file_buffer(associated_files[0].name).decode()

        # 将标签内容写入文件
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(labels_content)

        print(f"成功！标签已从 '{model_path}' 提取并保存到 '{export_path}'。")
        print("\n--- 标签内容预览 ---")
        # 打印前5个标签作为预览
        for i, line in enumerate(labels_content.splitlines()):
            if i < 5:
                print(line)
            else:
                break
        if len(labels_content.splitlines()) > 5:
            print("...")


    except Exception as e:
        print(f"提取标签时发生错误: {e}")
        print("\n请确认：")
        print(f"1. 文件路径 '{model_path}' 是否正确。")
        print("2. 该模型是否真的包含元数据。")

# --- 使用说明 ---
# 1. 将你的原版 .tflite 文件和这个 python 脚本放在同一个文件夹下。
# 2. 修改下面的 'your_original_model.tflite' 为你的实际文件名。
original_model_file = 'yolov10n_int8.tflite'

# 检查文件是否存在
if not os.path.exists(original_model_file):
    print(f"错误：找不到文件 '{original_model_file}'。请确保文件名正确且文件与脚本在同一目录下。")
else:
    extract_labels(original_model_file)