import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cow vs Buffalo Classifier",
    page_icon="🐄",
    layout="centered"
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# CLASS NAMES
# IMPORTANT:
# Keep the same order used during training
# =========================================================

class_names = ["Buffalo", "Cow"]

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    # -----------------------------------------------------
    # IMPORTANT:
    # Replace this section with the SAME model architecture
    # that you used while training.
    # -----------------------------------------------------

    model = torch.load(
        "model.pth",
        map_location=device,
        weights_only=False
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()

# =========================================================
# TEST TRANSFORM
# SAME TRANSFORM USED DURING TRAINING
# =========================================================

test_transforms = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# TITLE
# =========================================================

st.title("🐄 Cow vs Buffalo Classifier")

st.write(
    "Upload an image of a cow or buffalo "
    "to test the trained deep learning model."
)

st.divider()

# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    # -----------------------------------------------------
    # Apply transformations
    # -----------------------------------------------------

    input_tensor = test_transforms(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move to device
    input_batch = input_batch.to(device)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(input_batch)

        # For classification model
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

    # -----------------------------------------------------
    # Get result
    # -----------------------------------------------------

    predicted_class = class_names[predicted_idx.item()]

    confidence_percentage = confidence.item() * 100

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    st.divider()

    st.subheader("Prediction")

    if predicted_class == "Cow":

        st.success(
            f"🐄 Prediction: {predicted_class}"
        )

    else:

        st.info(
            f"🐃 Prediction: {predicted_class}"
        )

    st.metric(
        "Confidence",
        f"{confidence_percentage:.2f}%"
    )

    st.progress(
        confidence.item()
    )

    # -----------------------------------------------------
    # Show probabilities
    # -----------------------------------------------------

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = probabilities[0][i].item() * 100

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            probabilities[0][i].item()
        )

else:

    st.info(
        "👆 Upload a Cow or Buffalo image to start prediction."
    )