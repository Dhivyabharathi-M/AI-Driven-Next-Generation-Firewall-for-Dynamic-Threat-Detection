"""
Training Script for Firewall DNN Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import model
import sys
sys.path.append('model')
from model import FirewallDNN

class NetworkTrafficDataset(Dataset):
    """Custom Dataset for Network Traffic Data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.LongTensor(y.values if isinstance(y, pd.Series) else y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FirewallTrainer:
    def __init__(self, model_dir='model', data_dir='data/processed'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
    
    def load_data(self):
        """Load processed training and test data"""
        print("Loading data...")
        
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train_data.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test_data.csv'))
        
        # Separate features and labels
        X_train = train_df.drop('Label', axis=1)
        y_train = train_df['Label']
        
        X_test = test_df.drop('Label', axis=1)
        y_test = test_df['Label']
        
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(y_train.unique())}")
        
        return X_train, y_train, X_test, y_test
    
    def create_data_loaders(self, X_train, y_train, X_test, y_test, batch_size=256):
        """Create PyTorch DataLoaders"""
        train_dataset = NetworkTrafficDataset(X_train, y_train)
        test_dataset = NetworkTrafficDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return accuracy, precision, recall, f1, all_preds, all_labels
    
    def train(self, epochs=15, learning_rate=0.001, batch_size=256):
        """Complete training pipeline"""
        print("="*60)
        print("Training AI-Driven Firewall Model")
        print("="*60)
        
        X_train, y_train, X_test, y_test = self.load_data()
        
        train_loader, test_loader = self.create_data_loaders(
            X_train, y_train, X_test, y_test, batch_size
        )
        
        input_size = X_train.shape[1]
        num_classes = len(y_train.unique())
        
        model = FirewallDNN(input_size, num_classes).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"\nStarting training for {epochs} epochs...")
        train_losses = []
        train_accs = []
        
        for epoch in range(epochs):
            loss, acc = self.train_epoch(model, train_loader, criterion, optimizer)
            train_losses.append(loss)
            train_accs.append(acc)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
        
        print("\nEvaluating on test set...")
        accuracy, precision, recall, f1, preds, labels = self.evaluate(model, test_loader)
        
        print("\n" + "="*60)
        print("Final Test Results:")
        print("="*60)
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'firewall_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'num_classes': num_classes,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }, model_path)
        
        print(f"\nModel saved to: {model_path}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1)
        }
        
        with open(os.path.join(self.model_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(labels, preds, num_classes)
        
        return model, history
    
    def plot_confusion_matrix(self, y_true, y_pred, num_classes):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to: {self.model_dir}/confusion_matrix.png")

if __name__ == "__main__":
    trainer = FirewallTrainer()
    model, history = trainer.train(epochs=15, learning_rate=0.001)