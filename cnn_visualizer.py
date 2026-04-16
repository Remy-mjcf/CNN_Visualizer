# cnn_visualizer.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CNN Visualizer", layout="wide")

st.title("🧠 CNN Layer & Filter Visualizer (MNIST)")

# -------------------------------
# Load MNIST dataset
# -------------------------------
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., np.newaxis]  # add channel dim
x_test = x_test[..., np.newaxis]

# -------------------------------
# Define or load CNN model
# -------------------------------
@st.cache_resource
def load_or_train_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, validation_split=0.1, verbose=0)
    return model

model = load_or_train_model()

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Controls")
image_index = st.sidebar.slider("Pick a test image", 0, len(x_test)-1, 0)
layer_names = [layer.name for layer in model.layers if "conv" in layer.name]
selected_layer = st.sidebar.selectbox("Select a Conv layer", layer_names)

# -------------------------------
# Show original test image
# -------------------------------
st.subheader("Original MNIST Image")
col1, col2 = st.columns(2)

with col1:
    st.image(x_test[image_index].squeeze(), width=150, caption=f"Label: {y_test[image_index]}")

with col2:
    pred = np.argmax(model.predict(x_test[image_index][np.newaxis, ...], verbose=0))
    st.metric("Predicted Label", pred)

# -------------------------------
# Add Run button
# -------------------------------
if st.button("Run Visualization 🚀"):
    # Feature maps
    st.subheader(f"Feature maps from layer: {selected_layer}")

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(selected_layer).output
    )
    feature_maps = feature_model.predict(x_test[image_index][np.newaxis, ...])

    num_filters = feature_maps.shape[-1]
    cols = 8
    rows = int(np.ceil(num_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*2))
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_filters:
            ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
        ax.axis("off")
    st.pyplot(fig)

    # Learned filters
    st.subheader(f"Learned filters (kernels) from: {selected_layer}")
    weights, biases = model.get_layer(selected_layer).get_weights()

    # Normalize weights for display
    min_w, max_w = weights.min(), weights.max()
    weights = (weights - min_w) / (max_w - min_w)

    num_kernels = weights.shape[-1]
    k_rows = int(np.ceil(num_kernels / cols))

    fig_f, axes_f = plt.subplots(k_rows, cols, figsize=(15, k_rows*2))
    for i in range(k_rows * cols):
        ax = axes_f[i // cols, i % cols]
        if i < num_kernels:
            kernel = weights[:, :, 0, i]  # take first channel (MNIST is grayscale)
            ax.imshow(kernel, cmap="gray")
        ax.axis("off")
    st.pyplot(fig_f)
