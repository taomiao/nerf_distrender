import torch

from model_manager.model import Model
from model_optimization.optimization_pipeline import OptPipeline


class OptedModel:
    def __init__(self):
        self.optimization_pipeline = OptPipeline()
        self.opt_model = None

    def optimize(self, model):
        opt_model = self.optimization_pipeline.optimize(model)
        self.opt_model = opt_model
        return opt_model


if __name__ == "__main__":
    m = Model()
    m, model_class = m.load_from_path_and_name("model_test.model_1.model", "ModelFront")
    om = OptedModel().optimize(m)
    inp = torch.rand([1, 128]).cuda()
    res = om(inp)
    print(res)
