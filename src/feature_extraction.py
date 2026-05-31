import torch

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class FeatureExtractor(torch.nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet50_Weights.DEFAULT

        model = resnet50(weights=weights)

        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        return x