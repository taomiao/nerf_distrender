from unittest import TestCase
import torch
from seq_model.seq_model import ModelSeq


class TestModel(TestCase):
    def test_seq_model_without_opt(self):
        sm = ModelSeq().load_from_name("model_test", do_opt=False)
        inp = torch.rand([1, 128]).cuda()
        res = sm(inp)
        print(res)

    def test_seq_model_with_opt(self):
        sm = ModelSeq().load_from_name("model_test", do_opt=True)
        inp = torch.rand([1, 128]).cuda()
        res = sm(inp)
        print(res)
