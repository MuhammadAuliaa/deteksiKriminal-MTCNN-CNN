import os
import random
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import streamlit as st
from PIL import Image
detector = MTCNN()

def show_all_data_mtcnn(data_dir, categories, max_per_cat=3):
    detector = MTCNN()
    processed_images = []
    processed_labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        image_files = []
        for root, dirs, files in os.walk(category_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, f))

        if not image_files:
            st.warning(f"Tidak ada gambar di folder {category}.")
            continue

        st.write(f"### 🔍 {category.capitalize()} — MTCNN Detection")
        cols = st.columns(3)

        for i, img_path in enumerate(image_files[:max_per_cat]):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)

            img_with_box = img_rgb.copy()
            face_crop = None

            # --- gambar bounding box ---
            for res in results:
                x, y, w, h = res["box"]
                cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img_with_box,
                    f"{res['confidence']:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Simpan wajah crop pertama saja
                if face_crop is None:
                    face_crop = img_rgb[y:y + h, x:x + w]

            # --- tampilkan di Streamlit ---
            with cols[i % 3]:
                st.image(img_with_box, caption=f"{os.path.basename(img_path)} — bounding box", use_container_width=True)
                if face_crop is not None:
                    st.image(face_crop, caption=f"{category} (cropped face)", use_container_width=True)
                    processed_images.append(face_crop)
                    processed_labels.append(label)

    return processed_images, processed_labels

def preprocess_mtcnn_results(images):
    preprocessed = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.astype("float32") / 255.0
        preprocessed.append(normalized[..., np.newaxis])
    return np.array(preprocessed)

def show_preprocessed_samples(images, labels, categories, n=10):
    plt.figure(figsize=(15,3))
    indices = np.random.choice(len(images), min(n, len(images)), replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(1, n, i+1)
        plt.imshow(images[idx].squeeze(), cmap="gray")
        plt.title(categories[labels[idx]])
        plt.axis("off")

    st.pyplot(plt)

def predict_image_streamlit(model_path, img_file, categories, target_size=(64, 64), threshold=0.6):
    """
    Prediksi satu gambar yang diupload user lewat Streamlit.
    """
    # Load model
    loaded_model = load_model(model_path)

    # Baca gambar dari file upload (streamlit_uploaded_file)
    image = Image.open(img_file)
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Deteksi wajah
    results = detector.detect_faces(image_np)
    if len(results) != 1:
        st.warning(f"⚠️ {len(results)} wajah terdeteksi — proses testing dilewati.")
        return None

    x, y, w, h = results[0]['box']
    face = image_np[y:y+h, x:x+w]

    # Resize & grayscale
    face_resized = cv2.resize(face, target_size)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)

    # Normalisasi & reshape ke format CNN
    face_norm = face_gray.astype("float32") / 255.0
    face_input = np.expand_dims(face_norm, axis=(0, -1))  # (1, 64, 64, 1)

    # Prediksi
    pred = loaded_model.predict(face_input)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    # Visualisasi hasil
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(face_gray, cmap="gray")

    if confidence < threshold:
        label = "subject tidak terdeteksi"
        ax.set_title(f"{label}\n(confidence {confidence:.2f})")
        st.warning("❌ Subject tidak terdeteksi — confidence rendah.")
    else:
        label = categories[class_idx]
        ax.set_title(f"{label}\n(confidence {confidence:.2f})")
        st.success(f"✅ Predicted: **{label}** (confidence: {confidence:.2f})")

    ax.axis("off")
    st.pyplot(fig)

    return label, confidence