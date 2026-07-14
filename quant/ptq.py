import copy

import torch
import torch.nn as nn

from torch.ao.quantization import (
    get_default_qconfig,
    prepare,
    convert,
)

from torch.ao.quantization.qconfig import (
    float_qparams_weight_only_qconfig,
)


class PTQ:

    def __init__(self, backend: str = "qnnpack"):
        self.backend = backend

    def prepare(self, model: nn.Module):

        model = copy.deepcopy(model)

        model.eval()

        torch.backends.quantized.engine = self.backend

        default_qconfig = get_default_qconfig(self.backend)

        model.qconfig = default_qconfig

        # Embeddings require a different qconfig
        for module in model.modules():

            if isinstance(module, nn.Embedding):
                module.qconfig = float_qparams_weight_only_qconfig

        prepared_model = prepare(model)

        return prepared_model

    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str,
    ):

        model.to(device)
        model.eval()

        with torch.no_grad():

            for batch in dataloader:

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                batch.pop("labels")

                model(**batch)

        return model

    def convert(self, model: nn.Module):

        quantized_model = convert(model)

        return quantized_model

    def compress(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cpu",
    ):

        model = self.prepare(model)

        model = self.calibrate(
            model,
            dataloader,
            device,
        )

        model = self.convert(model)

        return model