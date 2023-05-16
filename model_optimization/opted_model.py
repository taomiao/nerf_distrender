import torch

from model_optimization.optimization_pipeline import OptPipeline


class OptedModel:
    def __init__(self):
        self.optimization_pipeline = OptPipeline()
        self.opt_model = None

    def optimize(self, model):
        opt_model = self.optimization_pipeline.optimize(model)
        self.opt_model = opt_model
        return opt_model

