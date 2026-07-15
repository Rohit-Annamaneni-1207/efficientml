from compression.context import PipelineContext


class CompressionPipeline:

    def __init__(self, model):
        self.model = model
        self.techniques = []

    def add(self, technique):
        self.techniques.append(technique)
        return self

    def apply(self, context: PipelineContext):
        model = self.model

        for technique in self.techniques:
            model = technique.apply(model, context)

        return model
