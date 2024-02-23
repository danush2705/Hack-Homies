import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import streamlit as st

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class names
class_names = ['benign', 'malignant', 'normal']  # Update with your actual class names

class CustomModel(nn.Module):
    def __init__(self, num_classes=3):  # Adjust num_classes to match the number of classes used during training
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 112 * 112, num_classes)  # Adjust output size to match number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_model(model_path):
    model = CustomModel(num_classes=3).to(device)  # Adjust num_classes to match the number of classes used during training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_disease(output):
    probabilities = torch.softmax(output, dim=1)
    predicted_class_index = torch.argmax(probabilities, dim=1).item()
    predicted_disease = class_names[predicted_class_index]
    return predicted_disease

def main():
    st.title("Disease Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Data augmentation and preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Apply transformations to input image
        input_image = transform(image)
        input_tensor = input_image.unsqueeze(0).to(device)

        # Load model
        model_path = r"C:\Users\svdan\OneDrive\Desktop\Bolt 2.0\best_model.pth"  # Provide the path to your saved model
        model = load_model(model_path)

        # Forward pass through the model
        output = model(input_tensor)

        # Get predicted disease
        predicted_disease = predict_disease(output)
        st.write("Predicted Disease:", predicted_disease)

if __name__ == "__main__":
    main()
