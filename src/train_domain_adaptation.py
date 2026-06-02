import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from feature_extraction import FeatureExtractor
from domain_adaptation import coral_loss

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

print("Using DRL sequence:", best_sequence)

source_dataset = ImageFolder(
    "../data_resplit/train",
    transform=get_transform_by_sequence(best_sequence)
)

target_dataset = ImageFolder(
    "../data_target/train",
    transform=get_baseline_transforms()
)

target_val_dataset = ImageFolder(
    "../data_target/val",
    transform=get_baseline_transforms()
)

source_loader = DataLoader(
    source_dataset,
    batch_size=32,
    shuffle=True
)

target_loader = DataLoader(
    target_dataset,
    batch_size=32,
    shuffle=True
)

target_val_loader = DataLoader(
    target_val_dataset,
    batch_size=32,
    shuffle=False
)

feature_extractor = FeatureExtractor().to(device)
classifier = nn.Linear(2048, 2).to(device)

optimizer = torch.optim.Adam(
    list(feature_extractor.parameters()) +
    list(classifier.parameters()),
    lr=1e-4
)

criterion = nn.CrossEntropyLoss()

lambda_da = 0.1
EPOCHS = 10
best_val_acc = 0

for epoch in range(EPOCHS):

    feature_extractor.train()
    classifier.train()

    running_loss = 0

    for source_batch, target_batch in zip(source_loader, target_loader):

        imgs_s, labels_s = source_batch
        imgs_t, _ = target_batch

        imgs_s = imgs_s.to(device)
        labels_s = labels_s.to(device)
        imgs_t = imgs_t.to(device)

        feat_s = feature_extractor(imgs_s)
        feat_t = feature_extractor(imgs_t)

        outputs = classifier(feat_s)

        loss_cls = criterion(outputs, labels_s)
        loss_da = coral_loss(feat_s, feat_t)

        loss = loss_cls + lambda_da * loss_da

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    feature_extractor.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in target_val_loader:

            images = images.to(device)
            labels = labels.to(device)

            features = feature_extractor(images)
            outputs = classifier(features)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {running_loss:.4f}")
    print(f"Target Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {
                "feature_extractor": feature_extractor.state_dict(),
                "classifier": classifier.state_dict()
            },
            "../models/best_drl_coral.pth"
        )

        print("Best DRL + CORAL model saved.")