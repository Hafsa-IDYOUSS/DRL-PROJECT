import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from baseline_model import get_model
from dataset import get_dataloaders


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, val_loader, test_loader = get_dataloaders(
    "../data/train",
    "../data/val",
    "../data/test"
)

model = get_model().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

EPOCHS = 10

best_val_acc = 0

for epoch in range(EPOCHS):

    # TRAINING
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    train_acc = correct / total

    # VALIDATION
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()

            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {running_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # SAVE BEST MODEL
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "../models/best_baseline.pth"
        )