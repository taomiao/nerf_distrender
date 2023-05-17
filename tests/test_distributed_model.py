from unittest import TestCase

import torch

from model_distributed.distributed_model import DistModel
from model_pipeline.piped_model import ModelPipe
from seq_model.model import Model
from seq_model.seq_model import ModelSeq


class TestDistModel(TestCase):

    def test_distributed_model(self):
        with torch.no_grad():
            sm = ModelSeq().load_from_name("model_test", do_opt=True)
            pm = ModelPipe().to_pipe(sm)
            dm = DistModel().to_distributed(pm)
            inp = torch.rand([1, 128]).cuda()
            res = dm(inp)
            print(res.to_here())