import torch

from distillation.loss import DistillationLoss


def main():

    criterion = DistillationLoss()

    student = torch.randn(
        8,
        2,
        requires_grad=True,
    )

    teacher = torch.randn(
        8,
        2,
    )

    labels = torch.randint(
        0,
        2,
        (8,),
    )

    loss = criterion(
        student,
        teacher,
        labels,
    )

    print(loss.item())

    loss.backward()

    print(student.grad is not None)


if __name__ == "__main__":
    main()