# Libraries
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Function to Load the model checkpoint, rebuild the architecture (including the classification head), and load the trained weights """

    checkpoint = torch.load(
        "best_model.pt",
        map_location=device,
        weights_only=True
    )

    model = models.resnet18(weights=None)

    fc_inputs = model.fc.in_features

    idx_to_class = checkpoint["idx_to_class"]
    num_classes = len(idx_to_class)

    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, idx_to_class


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# https://docs.streamlit.io/get-started/installation
# https://docs.streamlit.io/develop/api-reference/text

st.header("Satellite Image Wildfire Classification")
st.subheader("Upload a satellite image")

# https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
uploaded_image = st.file_uploader(
    "Upload a satellite image", type = ["jpg", "png", "jpeg"], label_visibility="collapsed"
)

model, idx_to_class = load_model()

def classify_image(model, image, idx_to_class):
    """Function for classifying user image"""

    user_image = Image.open(image).convert("RGB")

    input_tensor = image_transform(user_image).unsqueeze(0)

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        ret, prediction = torch.max(output.data, 1)
        class_name = idx_to_class[prediction.item()]

    return class_name

image_class = None

if uploaded_image is not None:
    try:
        st.image(uploaded_image, width=300)                                 # https://docs.streamlit.io/develop/api-reference/media/st.image
        image_class = classify_image(model, uploaded_image, idx_to_class)
        st.subheader(f"Prediction: {image_class}")
    except UnidentifiedImageError:
        st.error("Check that the uploaded file is in the jpg, png, or jpeg format.")

else:
    st.info("Upload a satellite image.")