from unittest import TestCase

from seq_model.model import Model


class TestModel(TestCase):

    def test_model(self):
        m = Model()
        m.load_from_path_and_name("model_test.model_1.model", "ModelFront")
