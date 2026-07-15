import torch.nn as nn

from quant.algorithms import SymmetricQuantizer
from quant.qat_layers import QATLinear
from quant.quantized_linear import QuantizedLinear


def convert_qat(model):

    for name, module in model.named_children():

        if isinstance(module, QATLinear):

            quantizer = SymmetricQuantizer()

            weight = module.linear.weight.detach()

            bias = (
                module.linear.bias.detach()
                if module.linear.bias is not None
                else None
            )

            q_weight = quantizer.quantize(weight)

            quantized = QuantizedLinear(
                q_weight=q_weight,
                bias=bias,
                scale=quantizer.scale,
                zero_point=quantizer.zero_point,
            )

            setattr(
                model,
                name,
                quantized,
            )

        else:

            convert_qat(module)

    return model