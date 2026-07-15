import torch.nn as nn

from quant.qat_layers import QATLinear


def inject_qat(
    model: nn.Module,
    num_bits: int = 8,
):

    for name, module in model.named_children():

        if isinstance(module, nn.Linear):

            setattr(
                model,
                name,
                QATLinear(
                    module,
                    num_bits=num_bits,
                ),
            )

        else:

            inject_qat(
                module,
                num_bits,
            )

    return model