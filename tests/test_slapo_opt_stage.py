import torch
import unittest
from slapo import create_schedule

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(128, 2)

    def forward(self, inp1, inp2):
        return self.fc(inp1) + self.fc(inp2)


class TestSlapoOptStage(unittest.TestCase):
    def test_optimize(self):
        m = MyModel()
        sch = create_schedule(m)
        print(sch)
