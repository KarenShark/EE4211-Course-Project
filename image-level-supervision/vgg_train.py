import torch

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    """
    Train a classification model and evaluate it on a validation set.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run training on (e.g., "cuda" or "cpu").
        epochs (int): Number of training epochs (default is 5).
    Prints:
        Training and validation loss and accuracy after each epoch.
    """
    
    for epoch in range(epochs):
        print(f"Training VGG epoch {epoch + 1}")
        # Training
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {running_loss:.3f}, Train Accuracy: {train_acc:.3f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()

        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc:.3f}")
    
    # Return final metrics for checkpoint saving
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': running_loss,
        'val_loss': val_loss
    }





