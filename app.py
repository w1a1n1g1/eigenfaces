import streamlit as st
import cv2
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import models, layers

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

def load_models():
    if os.path.exists('models/pca.pkl') and os.path.exists('models/knn.pkl') and os.path.exists('models/cnn_model.h5'):
        with open('models/pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('models/knn.pkl', 'rb') as f:
            knn = pickle.load(f)
        cnn_model = models.load_model('models/cnn_model.h5')
        cnn_feature_model = models.Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
        return pca, knn, cnn_model, cnn_feature_model, True
    return None, None, None, None, False

def save_models(pca, knn, cnn_model):
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    with open('models/knn.pkl', 'wb') as f:
        pickle.dump(knn, f)
    cnn_model.save('models/cnn_model.h5')

def train_models():
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

    X_train_lbp = np.array([extract_lbp_feature(img.reshape(128,128)*255) for img in images_flat])
    
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_lbp)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, labels)

    X_train_cnn = (images_flat * 255).reshape(-1, 128, 128, 1) / 255.0
    num_classes = len(np.unique(labels))

    cnn_model = models.Sequential([
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

    cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    cnn_model.fit(X_train_cnn, labels, epochs=15, batch_size=16, validation_split=0.2, verbose=0)
    
    save_models(pca, knn, cnn_model)
    cnn_feature_model = models.Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
    
    return pca, knn, cnn_model, cnn_feature_model, len(np.unique(labels))

st.set_page_config(page_title="人脸识别系统", page_icon="🔍", layout="wide")

st.title("🔍 人脸识别系统")

with st.sidebar:
    st.subheader("模型状态")
    status_placeholder = st.empty()
    
    pca, knn, cnn_model, cnn_feature_model, models_loaded = load_models()
    
    if models_loaded:
        status_placeholder.success("✅ 模型已加载")
    else:
        status_placeholder.warning("⚠️ 模型未训练")
        if st.button("开始训练模型"):
            with st.spinner("正在训练模型..."):
                pca, knn, cnn_model, cnn_feature_model, num_classes = train_models()
                status_placeholder.success(f"✅ 训练完成！\n识别人数: {num_classes}")
                models_loaded = True

col1, col2 = st.columns(2)

with col1:
    st.subheader("上传人脸图片")
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "pgm"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="上传的图片", use_column_width=True)
        
        if st.button("开始识别"):
            if not models_loaded:
                st.error("请先训练模型！")
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (128, 128))
                
                with col2:
                    st.subheader("识别结果")
                    
                    with st.spinner("LBP+Eigenfaces识别中..."):
                        lbp_feat = extract_lbp_feature(img_resized)
                        feat_pca = pca.transform([lbp_feat])
                        pred_label_lbp = knn.predict(feat_pca)[0]
                        st.success(f"LBP+Eigenfaces识别结果：身份 {pred_label_lbp}")
                    
                    with st.spinner("CNN识别中..."):
                        img_norm = img_resized / 255.0
                        img_input = img_norm.reshape(1, 128, 128, 1)
                        pred_prob = cnn_model.predict(img_input, verbose=0)
                        pred_label_cnn = np.argmax(pred_prob)
                        confidence = pred_prob[0][pred_label_cnn] * 100
                        st.success(f"CNN识别结果：身份 {pred_label_cnn} (置信度: {confidence:.1f}%)")
                    
                    feat = cnn_feature_model.predict(img_input, verbose=0)
                    st.info(f"CNN特征向量维度：{feat.shape[1]}维")

st.markdown("---")
st.subheader("关于")
st.write("""
本系统实现了两种人脸识别算法：
- **LBP+Eigenfaces+KNN**: 使用局部二值模式提取纹理特征，PCA降维后用KNN分类
- **CNN深度学习**: 使用卷积神经网络进行端到端人脸识别

数据集: ORL人脸数据集 (40人 × 10张图片 = 400张)
""")