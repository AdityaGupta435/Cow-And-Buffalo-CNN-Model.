import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms
from PIL import Image


st.set_page_config(
    page_title="Cow vs Buffalo AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def render_html(html):
    clean_html = "\n".join(
        line.lstrip()
        for line in html.splitlines()
    )

    st.markdown(
        clean_html,
        unsafe_allow_html=True
    )


# =========================================================
# CUSTOM CSS
# =========================================================

render_html("""
<style>

    /* Main background */
    .stApp {
        background: #f6f8fb;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #166534, #15803d);
        padding: 35px 40px;
        border-radius: 22px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(22, 101, 52, 0.15);
    }

    .hero h1 {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 16px;
        opacity: 0.9;
        margin: 0;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 20px rgba(0,0,0,0.04);
        height: 100%;
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Prediction */
    .prediction-card {
        background: linear-gradient(135deg, #ffffff, #f0fdf4);
        border: 1px solid #bbf7d0;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-name {
        color: #166534;
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .confidence {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
    }

    /* Info cards */
    .info-box {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
    }

    .info-number {
        font-size: 24px;
        font-weight: 750;
        color: #166534;
    }

    .info-text {
        font-size: 13px;
        color: #6b7280;
        margin-top: 3px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #f9fafb;
        border: 2px dashed #86efac;
        border-radius: 16px;
        padding: 10px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }

</style>
""")


# =========================================================
# MODEL
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=2):

        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 62 * 62,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cpu")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = torch.load(
        "model.pth",
        map_location="cpu",
        weights_only=False
    )

    def convert_quantized_linear(module):

        for name, child in list(module.named_children()):

            # Dynamic Quantized Linear
            if isinstance(
                child,
                torch.ao.nn.quantized.dynamic.Linear
            ):

                # Get quantized weight
                quantized_weight = child.weight()

                # Convert weight back to FP32
                weight = quantized_weight.dequantize()

                # Get bias
                bias = child.bias()

                # Create normal Linear layer
                new_linear = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=bias is not None
                )

                # Copy weights
                new_linear.weight.data.copy_(weight)

                # Copy bias
                if bias is not None:
                    new_linear.bias.data.copy_(
                        bias.detach().float()
                    )

                # Replace quantized layer
                module._modules[name] = new_linear

            else:
                convert_quantized_linear(child)

        return module

    # Quantized Linear → normal Linear
    model = convert_quantized_linear(model)

    # FP16 → FP32
    model = model.float()

    # Evaluation mode
    model.eval()

    return model


model = load_model()
# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Buffalo",
    "Cow"
]


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (250, 250)
    ),

    transforms.ToTensor()
])


# =========================================================
# HERO
# =========================================================

render_html("""
<div class="hero">

    <h1>🐄 Cow vs Buffalo AI</h1>

    <p>
        Intelligent image classification using a Convolutional Neural Network
    </p>

</div>
""")


# =========================================================
# INFO CARDS
# =========================================================

info1, info2, info3, info4 = st.columns(4)

with info1:

    render_html("""
    <div class="info-box">

        <div class="info-number">AI</div>

        <div class="info-text">
            Classification
        </div>

    </div>
    """)


with info2:

    render_html("""
    <div class="info-box">

        <div class="info-number">CNN</div>

        <div class="info-text">
            Deep Learning
        </div>

    </div>
    """)


with info3:

    render_html("""
    <div class="info-box">

        <div class="info-number">250×250</div>

        <div class="info-text">
            Input Image
        </div>

    </div>
    """)


with info4:

    render_html("""
    <div class="info-box">

        <div class="info-number">2</div>

        <div class="info-text">
            Classes
        </div>

    </div>
    """)


st.write("")


# =========================================================
# MAIN CONTENT
# =========================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT - UPLOAD
# =========================================================

with left:

    render_html("""
    <div class="card">

        <div class="card-title">
            📤 Upload Image
        </div>

        <div class="card-subtitle">
            Upload a clear image of a cow or buffalo
        </div>

    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file is None:

        st.info(
            "Supported formats: JPG, JPEG and PNG"
        )


# =========================================================
# RIGHT - PREVIEW
# =========================================================

with right:

    render_html("""
    <div class="card">

        <div class="card-title">
            🖼️ Image Preview
        </div>

        <div class="card-subtitle">
            Your uploaded image will appear here
        </div>

    </div>
    """)

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

    else:

        render_html("""
        <div style="
            height:280px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#f9fafb;
            border-radius:15px;
            color:#9ca3af;
            border:1px solid #e5e7eb;
        ">

            <div style="text-align:center">

                <div style="font-size:45px;">
                    🖼️
                </div>

                <div>
                    No image selected
                </div>

            </div>

        </div>
        """)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.write("")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.float()

    image_tensor = image_tensor.to(device)


    with st.spinner("AI is analyzing the image..."):

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()


    predicted_name = class_names[
        predicted_class
    ]


    # =====================================================
    # RESULT
    # =====================================================

    render_html(f"""
    <div class="prediction-card">

        <div class="prediction-label">
            AI Prediction
        </div>

        <div class="prediction-name">
            🐄 {predicted_name}
        </div>

        <div class="confidence">
            Confidence: {confidence * 100:.2f}%
        </div>

    </div>
    """)


    # =====================================================
    # CONFIDENCE BAR
    # =====================================================

    st.write("")

    st.progress(
        confidence,
        text=f"Model Confidence — {confidence * 100:.2f}%"
    )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.write("")

    st.markdown(
        "### 📊 Class Probabilities"
    )

    col1, col2 = st.columns(2)

    with col1:

        buffalo_prob = probabilities[
            0,
            0
        ].item()

        st.metric(
            "🐃 Buffalo",
            f"{buffalo_prob * 100:.2f}%"
        )

        st.progress(
            buffalo_prob
        )


    with col2:

        cow_prob = probabilities[
            0,
            1
        ].item()

        st.metric(
            "🐄 Cow",
            f"{cow_prob * 100:.2f}%"
        )

        st.progress(
            cow_prob
        )


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="footer">

    Cow vs Buffalo Classification • Powered by PyTorch CNN

</div>
""")
