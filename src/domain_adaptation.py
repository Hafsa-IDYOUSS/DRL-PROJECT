import torch


def compute_covariance(features):

    n = features.size(0)

    mean = torch.mean(
        features,
        dim=0,
        keepdim=True
    )

    features = features - mean

    covariance = (
        features.t() @ features
    ) / (n - 1)

    return covariance


def coral_loss(source, target):

    d = source.size(1)

    source_cov = compute_covariance(source)

    target_cov = compute_covariance(target)

    loss = torch.mean(
        (source_cov - target_cov) ** 2
    )

    loss = loss / (4 * d * d)

    return loss