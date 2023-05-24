from unittest import TestCase

from model_manager.model_manager import ModelManager
from seq_model.model import Model
import torch


class TestModelManager(TestCase):

    def test_model_load(self):
        mm = ModelManager()
        mm.load_models()
        m = mm.get_model_by_name("model_test")
        inp1 = torch.rand(128).cuda()
        inp2 = torch.rand(128).cuda()
        res = m(inp1, inp2)
        print(res)