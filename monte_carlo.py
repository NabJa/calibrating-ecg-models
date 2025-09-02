from typing import Tuple

import torch
from einops import reduce


def mc_resample(
    model, ecg, n_empty_samples, num_samples=100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs monte carlo (MC) resampling num_samples times."""
    model.train()  # Turn on dropout layers!

    all_class_preds, all_detailed_preds = [], []
    with torch.no_grad():
        for _ in range(num_samples):
            class_preds, detailed_preds = model(ecg, n_empty_samples)
            all_class_preds.append(class_preds)
            all_detailed_preds.append(detailed_preds)

    # Shape: [mc_samples, batch, label_logit]
    all_class_preds = torch.stack(all_class_preds)

    # Shape: [mc_samples, batch, label_logit, ecg_length]
    all_detailed_preds = torch.stack(all_detailed_preds)

    return all_class_preds, all_detailed_preds


def sample_entropy(predictions: torch.Tensor, dim=0) -> torch.Tensor:
    """Computes the sample binary cross entropy along a specified dimension."""
    bce = -(
        predictions * torch.log2(predictions)
        + (1 - predictions) * torch.log2(1 - predictions)
    )
    return bce.mean(dim=dim)


def summarize_prediction_confidence(predictions: torch.Tensor) -> dict:
    """
    Computes confidence as 1 - entropy and other basic statistics.
    Predictions should be mc samples with shape [mc_samples, batch, label].
    """

    assert predictions.ndim == 3, (
        "Predictions should have shape [mc_samples, batch, label]."
    )

    # Ensure predictions are in [0, 1] range.
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    # Ensures numeric stability.
    predictions = torch.clamp(predictions.detach().cpu(), min=1e-9, max=1 - 1e-9)

    # Compute sample entropy and confidence.
    entropy = reduce(predictions, "samples batch labels -> batch labels", reduction=sample_entropy)
    confidence = 1 - entropy.mean(dim=1)

    summary = {"confidence": confidence, "entropy": entropy, "mc_samples": predictions.shape[0]}

    for reduction in [torch.mean, torch.std, torch.var]:
        summary[reduction.__name__] = reduce(
            predictions, "samples batch labels -> batch", reduction=reduction
        )
        summary[f"label_{reduction.__name__}"] = reduce(
            predictions, "samples batch labels -> batch labels", reduction=reduction
        )

    return summary
