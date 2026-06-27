from typing import Dict, List, Optional
import torch
from tqdm import tqdm

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
        total_loss = 0
        
        for batch in self.train_loader:
            # move data to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Extract labels and not let model compute loss
            labels = batch.pop("labels")

            # Clear the gradients
            self.optimizer.zero_grad()

            #model outputs
            outputs = self.model(**batch)

            # Calculate loss
            loss = self.loss_fn(output.logits, labels)

            # Backward pass
            loss.backward()

            # Update params
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(self.train_loader)

        return avg_loss

    def validate(self) -> Optional[float]:
        self.model.eval()
        total_loss = 0

        if self.val_loader is None:
            return None

        with torch.no_grad():
            for batch in self.val_loader:

                batch = {k: v.to(self.device) for k, v in batch.items()}

                labels = batch.pop("labels")

                outputs = self.model(**batch)

                loss = self.loss_fn(outputs.logits, labels)

                total_loss += loss.item()

        avg_loss = total_loss/len(self.val_loader)
        return avg_loss
    

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        history = {"train_loss":[], "val_loss": []}

        for epoch in tqdm(range(epochs)):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        return history