from __future__ import annotations

try:
    from transformers import AutoModelForSequenceClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSequenceClassification = object  # type: ignore


def is_transformer_model(model) -> bool:
    return TRANSFORMERS_AVAILABLE and isinstance(
        model, AutoModelForSequenceClassification
    )


def freeze_layers(model, fraction_layers_to_finetune: float = 1.0) -> None:
    """
    Freeze a fraction of DistilBERT layers from the bottom.
    """
    total_layers = len(model.distilbert.transformer.layer)
    num_to_ft = int(fraction_layers_to_finetune * total_layers)
    num_to_freeze = total_layers - num_to_ft

    for layer in model.distilbert.transformer.layer[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
