import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

data_path = "face_dataset"
labels = []
images = []

for label, person in enumerate(os.listdir(data_path)):
    person_path = os.path.join(data_path, person)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (128, 128))
        images.append(img_resized)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)
images_flat = images.reshape(-1, 128*128) / 255.0
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.3, random_state=42)

print(f"数据集加载完成：共{len(images)}个样本，{len(np.unique(labels))}个身份")
print(f"训练集：{X_train.shape[0]}个样本，测试集：{X_test.shape[0]}个样本")

def extract_lbp_feature(image, radius=1, neighbors=8):
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape

    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center = image[i, j]
            binary = []
            for k in range(neighbors):
                angle = 2 * np.pi * k / neighbors
                x = i + int(radius * np.cos(angle))
                y = j + int(radius * np.sin(angle))
                binary.append(1 if image[x, y] >= center else 0)
            lbp_value = int(''.join(map(str, binary)), 2)
            lbp_image[i, j] = lbp_value

    hist, _ = np.histogram(lbp_image, bins=256, range=(0, 255))
    hist = hist / (np.sum(hist) + 1e-6)
    return hist

X_train_lbp = np.array([extract_lbp_feature(img.reshape(128,128)*255) for img in X_train])
X_test_lbp = np.array([extract_lbp_feature(img.reshape(128,128)*255) for img in X_test])

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_lbp)
X_test_pca = pca.transform(X_test_lbp)

print(f"\nPCA降维完成：原始LBP维度{X_train_lbp.shape[1]}，降维后{X_train_pca.shape[1]}")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)
acc_lbp = accuracy_score(y_test, y_pred)
print(f"基于LBP+Eigenfaces识别准确率：{acc_lbp:.2f}")

def predict_face_lbp(img_path, pca_model, knn_model):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128))
    lbp_feat = extract_lbp_feature(img_resized)
    feat_pca = pca_model.transform([lbp_feat])
    return knn_model.predict(feat_pca)[0]

test_img_path = "test_face.jpg"
if os.path.exists(test_img_path):
    pred_label = predict_face_lbp(test_img_path, pca, knn)
    print(f"待识别人脸身份标签(LBP)：{pred_label}")
else:
    print(f"警告：测试图片 {test_img_path} 不存在")

import tensorflow as tf
from tensorflow.keras import layers, models

X_train_cnn = (X_train * 255).reshape(-1, 128, 128, 1) / 255.0
X_test_cnn = (X_test * 255).reshape(-1, 128, 128, 1) / 255.0
num_classes = len(np.unique(labels))

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_cnn, y_train, epochs=20, batch_size=16, validation_split=0.2)

test_loss, test_acc_cnn = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\n基于CNN人脸识别准确率：{test_acc_cnn:.2f}")

cnn_feature_model = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)

def predict_face_cnn(img_path, feat_model, cls_model):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128,128))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1,128,128,1)
    feat = feat_model.predict(img_input, verbose=0)
    pred_prob = cls_model.predict(img_input, verbose=0)
    pred_label = np.argmax(pred_prob)
    return pred_label, feat

if os.path.exists(test_img_path):
    pred_label_cnn, cnn_feat = predict_face_cnn(test_img_path, cnn_feature_model, model)
    print(f"待识别人脸身份标签(CNN)：{pred_label_cnn}")
    print(f"CNN特征向量维度：{cnn_feat.shape[1]}维")
else:
    print(f"警告：测试图片 {test_img_path} 不存在")