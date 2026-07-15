from abc import ABC, abstractmethod
import torch


class BaseObserver(ABC):

    @abstractmethod
    def observe(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def get_range(self):
        pass

    @abstractmethod
    def calculate_qparams(self):
        pass

class MinMaxObserver(BaseObserver):

    def __init__(self, num_bits: int = 8):
        self.min_val = None
        self.max_val = None
        self.num_bits = num_bits
        self.qmax = 2 ** (self.num_bits - 1) - 1
        self.qmin = -(2 ** (self.num_bits - 1))

    def observe(self, tensor: torch.Tensor):

        tensor = tensor.detach().float()
        current_min = tensor.min()
        current_max = tensor.max()

        if self.min_val == None:
            self.min_val = current_min
            self.max_val = current_max

        else:
            self.min_val = torch.minimum(self.min_val, current_min)
            self.max_val = torch.maximum(self.max_val, current_max)

    def get_range(self):
        if self.min_val is None:
            raise ValueError(
                "Observer has not seen any tensors."
            )

        return self.min_val, self.max_val
    
    def calculate_qparams(self):

        max_abs = torch.max(
            torch.abs(self.min_val),
            torch.abs(self.max_val),
        )

        scale = max_abs / self.qmax

        if scale == 0:
            scale = torch.tensor(
                1.0,
                device=max_abs.device,
            )

        zero_point = torch.tensor(
            0.0,
            device=max_abs.device,
        )

        return scale, zero_point
    
if __name__ == "__main__":
    observer = MinMaxObserver()

    observer.observe(torch.tensor([-2., 4.]))
    observer.observe(torch.tensor([-6., 3.]))
    observer.observe(torch.tensor([-1., 8.]))

    print(observer.get_range())