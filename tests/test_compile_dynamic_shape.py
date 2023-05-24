import torch
from unittest import TestCase
import torch._dynamo
import torch._inductor
import torch._dynamo.config
import torch._inductor.config
import logging
torch._inductor.config.debug = True
torch._dynamo.config.verbose = True

import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/PJLAB/taomiao/PycharmProjects/distrender/tests/inductor_codecache"

class DynamicModel(torch.nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()

    def forward(self, x):
        y = x.sin()
        numel = y.shape[0] * y.shape[1]
        return y.view(numel)

class TestModel(TestCase):
    def test_dynamic_shape(self):
        m = DynamicModel()
        opt_m = torch.compile(m)
        inp1 = torch.rand([10, 128])
        res1 = opt_m(inp1)
        print(res1.shape)
        inp2 = torch.rand([100, 128])
        res2 = opt_m(inp2)
        print(res2.shape)