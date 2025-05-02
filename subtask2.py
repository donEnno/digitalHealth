# pyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Other imports
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# -------------------------------------------------------------------------------------# Constants

TRAIN_ROOT = 'technical_task/subtask_2/train'
VAL_ROOT = 'technical_task/subtask_2/val'

IMG_SIZE = 256

BATCH_SIZE = 16
N_EPOCHS = 1
LEARNING_RATE = 0.001

# -------------------------------------------------------------------------------------# Network and helper function

# Define a simple CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 Input channel (grayscale), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, sensitivity, specificity, and confusion matrix.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        accuracy (float): Accuracy of the model.
        sensitivity (float): Sensitivity (True Positive Rate).
        specificity (float): Specificity (True Negative Rate).
        cm (np.ndarray): Confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  
    specificity = tn / (tn + fp)  
    
    return accuracy, sensitivity, specificity, cm


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    """

    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations for each cell in the confusion matrix
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}', 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()
    

# -------------------------------------------------------------------------------------# Load and preprocess the dataset

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # One channel for grayscale
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create custom datasets using ImageFolder
trainset = ImageFolder(root=TRAIN_ROOT, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)

valset = ImageFolder(root=VAL_ROOT, transform=transform)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)

trainset.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
valset.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}


# -------------------------------------------------------------------------------------# Train the model

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.train()
for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print progress and stats
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch + 1}, Step {i + 1} / {len(trainloader)}, Loss: {running_loss / 10:.3f}', end='\r')
            running_loss = 0.0
    print('')

print('Finished Training')
print('')

# -------------------------------------------------------------------------------------# Evaluate the model on the validation set

net.eval()  # Set model to evaluation mode
all_preds, all_labels = [], []

with torch.no_grad():
    for data in valloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Convert to numpy arrays for easier manipulation
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy, sensitivity, specificity, cm = calculate_metrics(all_labels, all_preds)
print(f'Accuracy:    {accuracy:.4f}')
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
print('Plotting confusion matrix')

classes = trainset.classes
plot_confusion_matrix(cm, classes, title='Confusion Matrix')
