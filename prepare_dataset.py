import os
import shutil

source_dir = "."
target_dir = "face_dataset"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

folders = [f for f in os.listdir(source_dir) if f.startswith('s') and f[1:].isdigit()]

for folder in folders:
    person_id = int(folder[1:]) - 1
    target_folder = os.path.join(target_dir, f"person_{person_id}")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    src_folder = os.path.join(source_dir, folder)
    for img_name in os.listdir(src_folder):
        src_path = os.path.join(src_folder, img_name)
        if os.path.isfile(src_path):
            shutil.copy(src_path, target_folder)
            print(f"Copied: {src_path} -> {target_folder}")

print(f"\n数据集整理完成：{len(folders)}个人，已复制到 {target_dir}")