from unittest import TestCase

from model_manager.model_manager import ModelManager
from seq_model.model import Model


class TestModel(TestCase):

    def test_model_load(self):
        mm = ModelManager()
        mm.load_models()