from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from preprocessing import get_transforms


def get_dataloaders(train_dir, val_dir, test_dir):

    transform = get_transforms()

    train_dataset = ImageFolder(
        train_dir,
        transform=transform
    )

    val_dataset = ImageFolder(
        val_dir,
        transform=transform
    )

    test_dataset = ImageFolder(
        test_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )

    return train_loader, val_loader, test_loader