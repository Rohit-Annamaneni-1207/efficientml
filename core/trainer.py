from typing import Dict, List, Optional

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, train_loader, val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader


    def train_epoch(self) -> float:
        self.model.train()
        loss = 0
        # next_batch = next(iter(self.train_loader), None)
        return loss

    def validate(self) -> Optional[float]:
        self.model.eval()
        val_loss = 0
        # next_batch = next(iter(self.val_loader), None)
        return val_loss
    

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        pass