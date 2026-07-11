from abc import ABC, abstractmethod
import torch  # type: ignore[reportMissingImports]


class BaseQuantizer(ABC):
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits

        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

    @abstractmethod
    def quantize(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def dequantize(self, tensor: torch.Tensor):
        pass

class AffineQuantizer(BaseQuantizer):

    def __init__(self, num_bits: int = 8):
        super().__init__(num_bits)

        self.scale = None
        self.zero_point = None

    def calculate_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        xmin = tensor.min()
        xmax = tensor.max()

        self.scale = (xmax - xmin) / (self.qmax - self.qmin)
        
        return self.scale
    
    def calculate_zero_point(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            raise ValueError("Scale must be calculated before zero point.")

        xmin = tensor.min()

        self.zero_point = torch.round(self.qmin - (xmin / self.scale))
        
        self.zero_point = torch.clamp(
            self.zero_point,
            self.qmin,
            self.qmax
        )
        return self.zero_point
    
    def quantize(self, tensor: torch.Tensor):

        self.calculate_scale(tensor)
        self.calculate_zero_point(tensor)

        q_tensor = torch.round(
            tensor / self.scale
        ) + self.zero_point

        q_tensor = torch.clamp(
            q_tensor,
            self.qmin,
            self.qmax
        )

        return q_tensor.to(torch.int8)
    
    def dequantize(self, q_tensor: torch.Tensor):

        if self.scale is None or self.zero_point is None:
            raise ValueError(
                "Quantizer must be calibrated before dequantization."
            )

        return self.scale * (
            q_tensor.float() - self.zero_point
        )
    
    def reconstruction_error(self, tensor: torch.Tensor):

        q_tensor = self.quantize(tensor)
        reconstructed = self.dequantize(q_tensor)

        mse = torch.mean((tensor - reconstructed) ** 2)

        mae = torch.mean(torch.abs(tensor - reconstructed))

        return {
            "mse": mse.item(),
            "mae": mae.item()
        }
    

class SymmetricQuantizer(BaseQuantizer):

    def __init__(self, num_bits: int = 8):
        super().__init__(num_bits)

        self.scale = None
        self.zero_point = None

    def calculate_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        max_abs = torch.max(torch.abs(tensor))

        self.scale = max_abs/self.qmax
        
        return self.scale
    
    def calculate_zero_point(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            raise ValueError("Scale must be calculated before zero point.")

        self.zero_point = torch.tensor(
                0,
                dtype=self.scale.dtype,
                device=self.scale.device
            )
        return self.zero_point
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:

        self.calculate_scale(tensor)
        self.calculate_zero_point(tensor)

        q_tensor = torch.round(
            tensor / self.scale
        ) + self.zero_point

        q_tensor = torch.clamp(
            q_tensor,
            self.qmin,
            self.qmax
        )

        return q_tensor.to(torch.int8)
    
    def dequantize(self, q_tensor: torch.Tensor) -> torch.Tensor:

        if self.scale is None or self.zero_point is None:
            raise ValueError(
                "Quantizer must be calibrated before dequantization."
            )

        return self.scale * (
            q_tensor.float() - self.zero_point
        )
    
    def reconstruction_error(self, tensor: torch.Tensor) -> dict:

        q_tensor = self.quantize(tensor)
        reconstructed = self.dequantize(q_tensor)

        mse = torch.mean((tensor - reconstructed) ** 2)

        mae = torch.mean(torch.abs(tensor - reconstructed))

        return {
            "mse": mse.item(),
            "mae": mae.item()
        }