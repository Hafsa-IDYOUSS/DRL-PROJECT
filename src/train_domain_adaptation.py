import torch
import torch.nn as nn

from feature_extraction import FeatureExtractor
from dataset import get_dataloaders
from domain_adaptation import coral_loss

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, val_loader, test_loader = get_dataloaders(
    "../data/train",
    "../data/val",
    "../data/test"
)

# FEATURE EXTRACTOR
feature_extractor = FeatureExtractor().to(device)

# CLASSIFIER
classifier = nn.Linear(2048, 2).to(device)

optimizer = torch.optim.Adam(

    list(feature_extractor.parameters()) +
    list(classifier.parameters()),

    lr=1e-4
)

criterion = nn.CrossEntropyLoss()

lambda_da = 0.1

EPOCHS = 5

for epoch in range(EPOCHS):

    feature_extractor.train()
    classifier.train()

    running_loss = 0

    for (source_batch, target_batch) in zip(train_loader, val_loader):

        imgs_s, labels_s = source_batch
        imgs_t, _ = target_batch

        imgs_s = imgs_s.to(device)
        imgs_t = imgs_t.to(device)

        labels_s = labels_s.to(device)

        # FEATURE EXTRACTION
        feat_s = feature_extractor(imgs_s)
        feat_t = feature_extractor(imgs_t)

        # CLASSIFICATION
        outputs = classifier(feat_s)

        loss_cls = criterion(
            outputs,
            labels_s
        )

        # CORAL LOSS
        loss_da = coral_loss(
            feat_s,
            feat_t
        )

        # TOTAL LOSS
        loss = loss_cls + lambda_da * loss_da

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss: {running_loss:.4f}"
    )

torch.save(

    {
        "feature_extractor":
            feature_extractor.state_dict(),

        "classifier":
            classifier.state_dict()
    },

    "../models/best_coral.pth"
)

print("CORAL model saved!")