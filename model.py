"""
PyTorch Deep Neural Network Model for Network Intrusion Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FirewallDNN(nn.Module):
    """
    Deep Neural Network for Network Traffic Classification
    Architecture: Input -> 256 -> 128 -> 64 -> Output
    """
    def __init__(self, input_size, num_classes):
        super(FirewallDNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Layer 1: Input -> 256
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        # Layer 2: 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 3: 128 -> 64
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.fc4 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        
        return x

class FirewallCNN(nn.Module):
    """
    Convolutional Neural Network for Network Traffic Classification
    Alternative architecture using 1D convolutions
    """
    def __init__(self, input_size, num_classes):
        super(FirewallCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Reshape input to (batch, 1, features) for 1D convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flat_size = 128 * (input_size // 4)
        
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Reshape for 1D convolution
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x