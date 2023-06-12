import unittest
from model_optimization.fxgraph_opt_stage import FXGraphOptStage
from model_manager.model_manager import ModelManager
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(128, 2)

    def forward(self, inp1, inp2):
        return self.fc(inp1) + self.fc(inp2)
class TestFXGraphOptStage(unittest.TestCase):
    def test_optimize(self):
        mm = ModelManager()
        mm.load_models()
        opt_stage = FXGraphOptStage()

        m = mm.get_model_by_name("model_test")
        inp1 = torch.rand(128).cuda()
        inp2 = torch.rand(128).cuda()
        m(inp1, inp2)
        # m = MyModel()
        # opt_stage.optimize(m)

