import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from torch.nn.functional import softmax

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the class names
class_names = ['squamous.cell.carcinoma', 'normal', 'large.cell.carcinoma', 'adenocarcinoma','MalignantCases','BenginCases']

def load_model(model_path):
    # Define the model architecture
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(64 * 112 * 112, len(class_names))  # Adjust output size to match number of classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Initialize the model
    model = CustomModel().to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(image, model):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image tensor to the same device as the model
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        probs = softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class, confidence.item()

def main(image):
    # Load the trained model
    model_path = r"C:\Users\svdan\OneDrive\Desktop\Bolt 2.0\Trained model\trained_model.pth"
    model = load_model(model_path)

    # Classify the image
    predicted_class, confidence = predict_image(image, model)
    st.write("Predicted class:", predicted_class)
    st.write("Confidence:", confidence*100)

if __name__ == "__main__":
    st.title("Lung Cancer Image Classifier")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        main(image)
