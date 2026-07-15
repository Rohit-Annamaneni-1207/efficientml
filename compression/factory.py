from lora.inject import inject_lora
from quant.inject_qat import inject_qat


def apply_compression(
    model,
    method=None,
    **kwargs,
):

    if method is None:
        return model

    if method.lower() == "lora":
        return inject_lora(
            model,
            **kwargs,
        )

    if method.lower() == "qat":
        return inject_qat(
            model,
            **kwargs,
        )

    raise ValueError(
        f"Unknown compression method: {method}"
    )
