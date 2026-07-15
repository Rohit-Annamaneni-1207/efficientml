from compression.base import CompressionTechnique
from quant.convert_qat import convert_qat
from quant.inject_qat import inject_qat


class QAT(CompressionTechnique):

    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits

    def apply(self, model, context):
        model = inject_qat(
            model,
            num_bits=self.num_bits,
        )
        model.to(context.device)

        trainer = context.trainer
        trainer.model = model
        trainer.train_loader = context.train_loader
        trainer.val_loader = context.val_loader
        trainer.device = context.device
        trainer.fit(context.epochs)

        model = convert_qat(model)
        trainer.model = model

        return model

    def __repr__(self):
        return f"QAT(num_bits={self.num_bits})"
