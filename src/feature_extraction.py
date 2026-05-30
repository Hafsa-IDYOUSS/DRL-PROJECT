import torch
from torchvision import models


class FeatureExtractor(torch.nn.Module):

    def __init__(self):

        super().__init__()

        model = models.resnet50(pretrained=True)

        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):

        x = self.features(x)

        return x.squeeze()