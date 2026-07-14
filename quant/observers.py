from abc import ABC, abstractmethod
import torch


class BaseObserver(ABC):

    @abstractmethod
    def observe(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def get_range(self):
        pass

class MinMaxObserver(BaseObserver):

    def __init__(self):
        self.min_val = None
        self.max_val = None

    def observe(self, tensor: torch.Tensor):
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
    
if __name__ == "__main__":
    observer = MinMaxObserver()

    observer.observe(torch.tensor([-2., 4.]))
    observer.observe(torch.tensor([-6., 3.]))
    observer.observe(torch.tensor([-1., 8.]))

    print(observer.get_range())