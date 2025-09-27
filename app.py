import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# Load your trained model
# -----------------------------
model = torch.load("vgg16_full_model.pth", map_location="cpu")  # if you saved full model
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class labels
classes = ["COVID-19", "Viral Pneumonia", "Normal"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 COVID-19 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to predict COVID-19, Viral Pneumonia, or Normal.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: **{classes[predicted.item()]}**")
