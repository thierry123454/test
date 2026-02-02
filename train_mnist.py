"""
MNIST MLP Training Script with Reproducible Experiments

To reproduce experiments:

Baseline (no regularization):
    python train_mnist.py --exp_name baseline

Ablation (with dropout):
    python train_mnist.py --exp_name dropout_0.3 --dropout 0.3

Results are logged to:
    - runs/<exp_name>.jsonl (per-epoch metrics)
    - runs/<exp_name>_summary.json (final summary)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
import numpy as np
import argparse
import json
import os
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MLP on MNIST')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
parser.add_argument('--exp_name', type=str, default='baseline', help='Experiment name for logging')
args = parser.parse_args()

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Create runs directory if it doesn't exist
os.makedirs('runs', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'Experiment name: {args.exp_name}')
print(f'Weight decay: {args.weight_decay}')
print(f'Dropout: {args.dropout}')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 2
val_split = 0.2

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Train/validation split
train_size = int((1 - val_split) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# MLP Model
class MLP(nn.Module):
    def __init__(self, dropout=0.0):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = MLP(dropout=args.dropout).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Logging function
def log_epoch_metrics(exp_name, epoch, train_loss, val_acc, lr):
    """Save epoch metrics to JSONL file"""
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'lr': lr,
        'timestamp': datetime.now().isoformat()
    }
    
    log_file = f'runs/{exp_name}.jsonl'
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def save_summary(exp_name, config, final_test_acc, best_val_acc):
    """Save final summary to JSON file"""
    summary = {
        'exp_name': exp_name,
        'config': config,
        'final_test_accuracy': final_test_acc,
        'best_val_accuracy': best_val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = f'runs/{exp_name}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

# Training loop
print(f'\nStarting training for {num_epochs} epochs...\n')
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_accuracy = evaluate(model, val_loader, device)
    
    # Track best validation accuracy
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
    
    # Log metrics
    log_epoch_metrics(args.exp_name, epoch + 1, train_loss, val_accuracy, learning_rate)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# Final test accuracy
test_accuracy = evaluate(model, test_loader, device)
print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')

# Save summary
config = {
    'weight_decay': args.weight_decay,
    'dropout': args.dropout,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs
}
save_summary(args.exp_name, config, test_accuracy, best_val_acc)
print(f'\nLogs saved to runs/{args.exp_name}.jsonl')
print(f'Summary saved to runs/{args.exp_name}_summary.json')
