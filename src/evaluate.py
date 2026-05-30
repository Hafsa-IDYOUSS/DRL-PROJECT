import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def evaluate(model, dataloader, device):

    model.eval()

    predictions = []
    labels_list = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        labels_list,
        predictions
    )

    print(f"\nAccuracy: {accuracy:.4f}\n")

    print(
        classification_report(
            labels_list,
            predictions
        )
    )

    print(
        confusion_matrix(
            labels_list,
            predictions
        )
    )