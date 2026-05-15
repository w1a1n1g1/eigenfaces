import cv2
import numpy as np

np.random.seed(99)
base_pattern = np.random.rand(128, 128) * 255
noise = np.random.normal(0, 15, (128, 128))
face_img = base_pattern + noise
face_img = np.clip(face_img, 0, 255).astype(np.uint8)

eye_region = np.random.randint(20, 60, size=4)
face_img[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]] = np.random.randint(10, 80, size=face_img[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]].shape)

mouth_region = np.random.randint(80, 110, size=4)
face_img[mouth_region[0]:mouth_region[1], mouth_region[2]:mouth_region[3]] = np.random.randint(20, 100, size=face_img[mouth_region[0]:mouth_region[1], mouth_region[2]:mouth_region[3]].shape)

cv2.imwrite("test_face.jpg", face_img)
print("测试图片已创建：test_face.jpg")