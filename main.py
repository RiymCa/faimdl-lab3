import torch
from torch import nn
from models.customNet import CustomNet
from train import train
from eval import validate
from data.dataloader import train_loader, val_loader

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CustomNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# uncomment for longer version
# num_epochs = 10
num_epochs = 1
print(f"Starting training for {num_epochs} epoch over the full dataset...")

for epoch in range(1, num_epochs + 1):
    # Train step (Notice we added 'device' here to fix the previous crash)
    train(epoch, model, train_loader, criterion, optimizer, device)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion, device)

print(f'Final validation accuracy: {val_accuracy:.2f}%')


