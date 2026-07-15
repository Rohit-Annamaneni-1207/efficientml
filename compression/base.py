from abc import ABC, abstractmethod

import torch.nn as nn

from compression.context import PipelineContext


class CompressionTechnique(ABC):

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        context: PipelineContext,
    ) -> nn.Module:
        raise NotImplementedError
