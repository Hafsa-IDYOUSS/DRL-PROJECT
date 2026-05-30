import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


def get_model():

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Linear(num_features, 256),

        nn.ReLU(),

        nn.Dropout(0.3),

        nn.Linear(256, 2)

    )

    return model