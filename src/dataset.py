from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_dataloaders(
    train_dir,
    val_dir,
    test_dir,
    train_transform,
    eval_transform,
    batch_size=32
):

    train_dataset = ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = ImageFolder(
        test_dir,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader