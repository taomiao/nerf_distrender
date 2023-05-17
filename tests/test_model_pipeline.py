from unittest import TestCase

import torch

from model_pipeline.piped_model import ModelPipe
from seq_model.seq_model import ModelSeq


class TestModelPipe(TestCase):

    def test_model_pipeline(self):
        sm = ModelSeq().load_from_name("model_test", do_opt=True)
        pm = ModelPipe().to_pipe(sm)
        inp = torch.rand([1, 128]).cuda()
        res = pm(inp)
        print(res.to_here())