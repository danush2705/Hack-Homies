import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import streamlit as st

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the class names for each model
class_names_brain_tumor = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_names_breast_ultrasound = ['benign', 'malignant', 'normal']

# Function to load the model based on the selected option
def load_model(model_name):
    model_path = ""
    class_names = []

    if model_name == "Brain Tumor MRI":
        model_path = r"C:\Users\svdan\OneDrive\Desktop\Bolt 2.0\Brain Tumor MRI Dataset\Trained model\glioma_model.pth"
        class_names = class_names_brain_tumor
    elif model_name == "Breast Ultrasound":
        model_path = r"C:\Users\svdan\OneDrive\Desktop\Bolt 2.0\best_model.pth"
        class_names = class_names_breast_ultrasound

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
    return model, class_names

# Function to preprocess and predict the image
def predict_image(image, model, class_names):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image tensor to the same device as the model
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image) 
        predicted_index = torch.argmax(outputs, 1).item()
        predicted_class = class_names[predicted_index]
        return predicted_class

def main():
    st.title("Image Classification")

    # Options for different models
    model_option = st.selectbox("Select Model", ["Brain Tumor MRI", "Breast Ultrasound"])

    # Load the selected model
    model, class_names = load_model(model_option)

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify the image
        predicted_class = predict_image(image, model, class_names)
        st.write("Predicted Class:", predicted_class)

if __name__ == "__main__":
    main()
