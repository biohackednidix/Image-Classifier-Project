import argparse
import torch
from torch import nn, optim
import torchvision.models as models
from torchvision import datasets, transforms
import json
import os

# Load and process data
def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=False)
    }

    return image_datasets, dataloaders

# Build the model
def build_model(arch='vgg16', hidden_units=512):
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  # Freeze pre-trained weights

    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

# Train the model
def train_model(model, dataloaders, criterion, optimizer, epochs=5, device='cpu'):
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                probs = torch.exp(outputs)
                top_p, top_class = probs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train Loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation Loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# Save checkpoint
def save_checkpoint(model, save_dir, arch, hidden_units):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'classifier': model.classifier
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

# Argument parsing
parser = argparse.ArgumentParser(description="Train a neural network on a dataset")

parser.add_argument('data_dir', help="Path to dataset")
parser.add_argument('--save_dir', default='./', help="Directory to save the model checkpoint")
parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'vgg13'], help="Choose model architecture")
parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units in classifier")
parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training")

args = parser.parse_args()

# Load data
image_datasets, dataloaders = load_data(args.data_dir)

# Build model
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model = build_model(args.arch, args.hidden_units)

# Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train model
train_model(model, dataloaders, criterion, optimizer, args.epochs, device)

# Save checkpoint
save_checkpoint(model, args.save_dir, args.arch, args.hidden_units)
print(f"Model saved to {args.save_dir}/checkpoint.pth")
