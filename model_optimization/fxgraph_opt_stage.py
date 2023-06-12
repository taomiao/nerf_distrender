import torch

from model_optimization.optimization_pipeline import OptStage

class FXGraphOptStage(OptStage):
    def optimize(self, model: torch.nn.Module):
        print(torch.fx.symbolic_trace(model))