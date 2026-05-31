import json

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from baseline_model import get_model
from dataset import get_dataloaders

from preprocessing import (
    get_baseline_transforms,
    get_transform_by_sequence
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open("../results/rl_policy.json", "r") as f:
    policy = json.load(f)

best_sequence = policy["best_sequence_actions"]

print("Using DRL-selected sequence:", best_sequence)

train_loader, val_loader, test_loader = get_dataloaders(
    "../data_resplit/train",
    "../data_resplit/val",
    "../data_resplit/test",
    train_transform=get_transform_by_sequence(best_sequence),
    eval_transform=get_baseline_transforms(),
    batch_size=32
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

    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "../models/best_drl.pth"
        )

        print("Best DRL-CNN model saved.")