from typing import Dict, List, Optional

import torch
from tqdm import tqdm


class DistillationTrainer:

    def __init__(
        self,
        teacher,
        student,
        optimizer,
        loss_fn,
        device,
        train_loader,
        val_loader=None,
    ):
        self.teacher = teacher
        self.student = student

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Teacher is never trained
        self.teacher.eval()

        for param in teacher.parameters():
            param.requires_grad = False

    def train_epoch(self) -> float:

        self.student.train()

        total_loss = 0

        print("Starting distillation training loop")

        for batch in tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        ):

            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
            }

            labels = batch.pop("labels")

            self.optimizer.zero_grad()

            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = self.teacher(**batch)

            # Student forward pass
            student_outputs = self.student(**batch)

            loss = self.loss_fn(
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def validate(self) -> Optional[float]:

        if self.val_loader is None:
            return None

        self.student.eval()

        total_loss = 0

        with torch.no_grad():

            for batch in tqdm(
                self.val_loader,
                desc="Validation",
                leave=False,
            ):

                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }

                labels = batch.pop("labels")

                teacher_outputs = self.teacher(**batch)

                student_outputs = self.student(**batch)

                loss = self.loss_fn(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    labels,
                )

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss

    def fit(
        self,
        epochs: int,
    ) -> Dict[str, List[float]]:

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in tqdm(range(epochs)):

            train_loss = self.train_epoch()

            val_loss = self.validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}",
                end="",
            )

            if val_loss is not None:
                print(
                    f" | Val Loss: {val_loss:.4f}"
                )
            else:
                print()

        return history