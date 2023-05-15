import os.path

import torch
import importlib
import sys
from config import models_repo

class Model:
    def __init__(self):
        sys.path.append(models_repo)

    @staticmethod
    def load_from_path_and_name(path, cls_name=None):
        mod = __import__(path, fromlist=["load_model"])
        load_model = getattr(mod, "load_model")
        model = load_model()
        m_class = None
        if cls_name:
            m_class = getattr(mod, cls_name)
        return model, m_class

if __name__=="__main__":
    m = Model()
    m.load_from_path_and_name("model_test.model_1.model", "ModelFront")