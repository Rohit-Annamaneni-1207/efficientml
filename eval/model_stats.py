def parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100 * trainable / total,
    }


def model_size(model):
    size = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    return size / (1024 ** 2)