import os

# 指定要读取的文件夹路径
folder_path = r"D:\ultralytics-main\coco2017\train2017"  # 替换为你的文件夹路径
output_file = "train.txt"  # 存储结果的txt文件

# folder_path = r"D:\ultralytics-main\coco2017\val2017"  # 替换为你的文件夹路径
# output_file = "val.txt"  # 存储结果的txt文件

# 获取文件夹中所有文件的绝对路径
file_paths = [
    os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
]

# 将文件路径按行写入txt文件
with open(output_file, "w") as f:
    f.writelines(file_path + "\n" for file_path in file_paths)

print(f"文件路径已存储在 {output_file} 中。")
