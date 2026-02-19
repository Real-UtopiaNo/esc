class PTQBase:
    def __init__(self, model, **kwargs):
        self.model = model

    def run(self):
        """Perform PTQ and return quantized model"""
        raise NotImplementedError