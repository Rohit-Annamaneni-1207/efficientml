from compression.base import CompressionTechnique
from quant.ptq import PTQ as PTQCompressor


# class PTQ(CompressionTechnique):

#     def __init__(self, backend: str = "qnnpack"):
#         self.backend = backend
#         self.compressor = PTQCompressor(
#             backend=backend,
#         )

#     def apply(self, model, context):

#         calibration_loader = (
#             context.calibration_loader
#             or context.val_loader
#             or context.train_loader
#         )

#         # PTQ only works on CPU
#         model = model.cpu()

#         # Update the context so downstream code (e.g. benchmarking)
#         # knows the model now lives on CPU.
#         context.device = "cpu"

#         model = self.compressor.prepare(model)

#         model = self.compressor.calibrate(
#             model,
#             calibration_loader,
#             context.device,
#         )

#         model = self.compressor.convert(model)

#         return model

#     def __repr__(self):
#         return f"PTQ(backend={self.backend!r})"

class PTQ(CompressionTechnique):

    def __init__(self):
        self.compressor = PTQCompressor()

    def apply(self, model, context):

        calibration_loader = (
            context.calibration_loader
            or context.val_loader
            or context.train_loader
        )

        # PTQ only works on CPU
        model = model.cpu()

        # Update the context so downstream code (e.g. benchmarking)
        # knows the model now lives on CPU.
        context.device = "cpu"

        return self.compressor.compress(
            model,
            context.calibration_loader,
            context.device,
        )

    def __repr__(self):
        return "DynamicPTQ()"