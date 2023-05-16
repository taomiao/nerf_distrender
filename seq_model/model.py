import sys

from config import config

models_repo = config["models_repo"]
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
