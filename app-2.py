import os

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "baseline_cnn.keras"  # local filename the model is stored/cached as

# Your model is 155MB, which is over GitHub's 100MB push limit, so instead of
# committing it to the repo the app downloads it once at startup (from your
# Google Drive file) and caches it on disk.
#
# IMPORTANT: the Drive file must be shared as "Anyone with the link" (Viewer)
# or the download below will fail silently and pull down an HTML error page
# instead of the model.
GDRIVE_FILE_ID = "1rllbEhoNmMH9QOI2qTDMLwGJTdn0fK9D"

IMG_SIZE = (224, 224)

# IMPORTANT: Keras' image_dataset_from_directory sorts class folder names
# alphabetically. This list assumes that order. Check the "Class names:"
# output your training notebook printed (train_ds.class_names) and edit
# this list if the order is different.
CLASS_NAMES = ["F", "M", "N", "Q", "S", "V"]

CLASS_DESCRIPTIONS = {
    "F": "Fusion beat",
    "M": "M-class beat",
    "N": "Normal beat",
    "Q": "Unknown / unclassifiable beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
}

st.set_page_config(page_title="ECG Beat Classifier", page_icon="🫀", layout="centered")

# ---------------------------------------------------------------------------
# Model loading (download once if needed, then cache for the app's lifetime)
# ---------------------------------------------------------------------------


def download_model():
    if os.path.exists(MODEL_PATH):
        return

    with st.spinner("Downloading model (first run only, ~155MB)..."):
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)

    # gdown writes an HTML error page instead of raising if the file isn't
    # publicly shared, so check the result actually looks like a model file.
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        st.error(
            "Model download failed. Check that the Google Drive file is "
            "shared as 'Anyone with the link' (Viewer), not restricted."
        )
        st.stop()


@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Preprocessing — mirrors the training pipeline's preprocess_grayscale step:
# resize -> RGB -> grayscale -> normalize to [0, 1]
# ---------------------------------------------------------------------------


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    tensor = tf.convert_to_tensor(array)
    gray = tf.image.rgb_to_grayscale(tensor)
    gray = tf.cast(gray, tf.float32) / 255.0
    return tf.expand_dims(gray, axis=0).numpy()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("ECG Beat Classifier")
st.write(
    "Upload an ECG trace image and the model will classify the beat type "
    "(F / M / N / Q / S / V)."
)

uploaded_file = st.file_uploader("ECG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG", use_container_width=True)

    with st.spinner("Running model..."):
        model = load_model()
        input_tensor = preprocess(image)
        probabilities = model.predict(input_tensor, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader(
        f"Prediction: {predicted_class} — {CLASS_DESCRIPTIONS.get(predicted_class, '')}"
    )
    st.write(f"Confidence: {confidence:.1%}")

    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probabilities)})

st.divider()
st.caption(
    "This tool is for educational/demonstration purposes only. It is not a "
    "medical device and should not be used to make clinical decisions."
)
