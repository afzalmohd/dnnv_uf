import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.model_selection import train_test_split

num_epochs = 15
saved_model_name = 'pytorch_model.onnx'


def save_model_as_onnx(model, input_sample, filename="model.onnx"):
    # Export the model to ONNX
    torch.onnx.export(
        model,                            # Model to be exported
        input_sample,                     # Example input to define input shape
        filename,                         # Filename to save the ONNX model
        export_params=True,               # Store the trained parameter weights
        opset_version=11,                 # ONNX version (can be adjusted based on requirements)
        do_constant_folding=True,         # Optimization to fold constants
        input_names=['input'],            # Names of input nodes
        output_names=['output'],          # Names of output nodes
        dynamic_axes={'input': {0: 'batch_size'},  # Dynamic axes for variable batch size
                      'output': {0: 'batch_size'}}
    )
    print(f"Model has been saved as {filename}")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)  # 1 input channel (grayscale), 16 output channels
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 24 * 24, 128)  # Adjust the input size of Linear layer accordingly
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=1)
        x = self.fc2(x)
        return x


def training(train_x, train_y, val_x, val_y, model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Convert data to PyTorch tensors and create dataloaders
    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(train_y, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_x, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(val_y, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Compute training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%", end=', ')

        # Validation step (optional)
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_accuracy = 100 * correct_val / total_val
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

    return model

def training_with_augmentation(train_x, train_y, val_x, val_y, model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    early_stopping_accuracy = 99.15

    # Define the data augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Random zoom between 80% to 120%
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_x, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Ensure inputs have the correct shape: (B, H, W, C) -> (B, C, H, W)
            inputs = inputs.permute(0, 3, 1, 2)  # Convert from (B, H, W, C) to (B, C, H, W)
            inputs = torch.stack([transform(image) for image in inputs])  # Apply transformations

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%", end=', ')

        # Validation step (optional)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            validation_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {validation_accuracy:.2f}%')

        if validation_accuracy >= early_stopping_accuracy:
            print(f"Early stopping triggered. Validation accuracy has reached {validation_accuracy:.2f}%")
            break

    return model





mnist = tf.keras.datasets.mnist

(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

x_train_full = x_train_full / 255.0
x_train_full = x_train_full.reshape(x_train_full.shape[0], 28, 28, 1)
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Instantiate the model
model = MyModel()

# trained_model = training(x_train, y_train, x_val, y_val, model, num_epochs=num_epochs)

trained_model = training_with_augmentation(x_train, y_train, x_val, y_val, model, num_epochs=num_epochs)

input_sample = torch.tensor(x_val[:1], dtype=torch.float32).permute(0, 3, 1, 2)  # Example input sample
save_model_as_onnx(trained_model, input_sample, filename=saved_model_name)

# print(trained_model)
