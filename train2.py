import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from PIL import Image

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation to apply to the images
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.data[idx]).convert('RGB')  # Convert to RGB in case of grayscale images
            target = self.targets[idx]

            if self.transform:
                image = self.transform(image)

            return image, target
        except Exception as e:
            print(f"Error loading image: {self.data[idx]}")
            print(e)
            return None, None

def main(training_directory):
    # Load the dataset
    data = []
    targets = []
    for label in os.listdir(training_directory):
        label_directory = os.path.join(training_directory, label)
        for image_file in os.listdir(label_directory):
            image_path = os.path.join(label_directory, image_file)
            data.append(image_path)
            targets.append(label)

    # Encode the target labels into integer indices
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(targets)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, targets, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(X_val, y_val, transform=test_transform)

    # Filter out None values (failed to load images)
    train_dataset = [(image, target) for image, target in train_dataset if image is not None]
    val_dataset = [(image, target) for image, target in val_dataset if image is not None]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define the model
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(64 * 112 * 112, len(label_encoder.classes_))  # Adjust output size to match the number of classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = CustomModel().to(device)

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    best_val_loss = float('inf')
    for epoch in range(50):  # Train for 50 epochs
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1}/{50}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step()

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    print('Finished Training')

if __name__ == '__main__':
    training_directory = r"C:\Users\svdan\OneDrive\Desktop\Bolt 2.0\Breast Ultrasound\Dataset_BUSI_with_GT"
    main(training_directory)
