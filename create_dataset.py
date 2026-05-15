import cv2
import numpy as np
import os

np.random.seed(42)

data_path = "face_dataset"
if not os.path.exists(data_path):
    os.makedirs(data_path)

num_persons = 5
images_per_person = 20

for person_id in range(num_persons):
    person_folder = os.path.join(data_path, f"person_{person_id}")
    os.makedirs(person_folder, exist_ok=True)
    
    base_pattern = np.random.rand(128, 128) * 255
    
    for img_id in range(images_per_person):
        noise = np.random.normal(0, 15, (128, 128))
        face_img = base_pattern + noise
        face_img = np.clip(face_img, 0, 255).astype(np.uint8)
        
        eye_region = np.random.randint(20, 60, size=4)
        face_img[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]] = np.random.randint(10, 80, size=face_img[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]].shape)
        
        mouth_region = np.random.randint(80, 110, size=4)
        face_img[mouth_region[0]:mouth_region[1], mouth_region[2]:mouth_region[3]] = np.random.randint(20, 100, size=face_img[mouth_region[0]:mouth_region[1], mouth_region[2]:mouth_region[3]].shape)
        
        img_path = os.path.join(person_folder, f"face_{img_id}.jpg")
        cv2.imwrite(img_path, face_img)
        print(f"Created: {img_path}")

print(f"\n数据集创建完成：{num_persons}个人，每人{images_per_person}张图片")