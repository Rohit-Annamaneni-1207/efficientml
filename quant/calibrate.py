import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from observers import MinMaxObserver


class Calibrator:

    def __init__(self, model):
        self.model = model
        self.observers = {}
        self.handles = []

    def attach_observers(self):

        for name, module in self.model.named_modules():

            if not isinstance(module, nn.Linear):
                continue

            observer = MinMaxObserver()
            self.observers[name] = observer

            def hook(module, inputs, output, obs=observer):
                obs.observe(output.detach())

            handle = module.register_forward_hook(hook)

            self.handles.append(handle)

    def calibrate(self,
    dataloader: DataLoader,
    device: str,
    ):

        self.attach_observers()

        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                batch.pop("labels")
                self.model(**batch)

        self.remove_observers()

        return self.observers
    
    def remove_observers(self):

        for handle in self.handles:
            handle.remove()

        self.handles.clear()