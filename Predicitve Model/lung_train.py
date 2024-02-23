import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main(training_directory):
    # Load the dataset
    train_dataset = ImageFolder(root=training_directory, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    total_batches = len(train_loader)

    # Define the model
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(64 * 112 * 112, len(train_dataset.classes))  # Adjust output size to match number of classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = CustomModel().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % total_batches == 0:  # Print at the end of each epoch
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total_batches))
                running_loss = 0.0

    print('Finished Training')
    
    # Create directory if it does not exist
    save_directory = "Trained model"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the trained model in the "Trained model" folder with a default file name
    model_path = os.path.join(save_directory, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved as '{model_path}'")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    training_directory = r"C:\Users\svdan\OneDrive\Desktop\Bolt 2.0\Lung Cancer\LungcancerDataSet\Data\train"
    main(training_directory)
