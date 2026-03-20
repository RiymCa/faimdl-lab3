import torch
from torch import nn
import wandb  # <-- Import wandb

from models.customNet import CustomNet
from train import train
from eval import validate
from data.dataloader import train_loader, val_loader

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize W&B Run
    wandb.init(
        project="tiny-imagenet-lab",  # The name of your project in the W&B dashboard
        name="customNet-run-1",       # Optional: give this specific run a name
        config={
            "learning_rate": 0.001,
            "epochs": 20,             # Update this to however many epochs you want
            "batch_size": 32,
            "momentum": 0.9,
            "architecture": "CustomNet",
            "dataset": "TinyImageNet"
        }
    )

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Notice we can pull the hyperparameters directly from wandb.config!
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=wandb.config.momentum)

    num_epochs = wandb.config.epochs
    print(f"Starting training for {num_epochs} epochs...")

    best_acc = 0

    for epoch in range(1, num_epochs + 1):
        # Run training
        t_loss, t_acc = train(epoch, model, train_loader, criterion, optimizer, device)
        
        # Run validation
        v_acc = validate(model, val_loader, criterion, device)
        
        # Save best accuracy
        best_acc = max(best_acc, v_acc)

        # 2. Log Metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": t_loss,
            "train_accuracy": t_acc,
            "val_accuracy": v_acc
        })

    print(f'Training complete. Best Validation Accuracy: {best_acc:.2f}%')
    
    # 3. Finish the run
    wandb.finish()