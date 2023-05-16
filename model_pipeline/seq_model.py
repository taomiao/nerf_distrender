import os
import sys

import torch

from config import models_repo
from model_manager.model import Model
from model_optimization.opted_model import OptedModel


class ModelSeq:
    def __init__(self):
        sys.path.append(models_repo)
        self.models = []
        self.seq_model = None

    def load_from_name(self, name, do_opt=False):
        path = os.path.join(models_repo, name)
        dirs = sorted(os.listdir(path))
        for d_name in dirs:
            d_path = ".".join([name, d_name, "model"])
            if d_name.startswith("__"):
                continue
            print(d_path)
            model, model_class = Model.load_from_path_and_name(d_path)
            if do_opt:
                model = OptedModel().optimize(model)
            self.models.append(model)
        seq_model = torch.nn.Sequential(*self.models)
        self.seq_model = seq_model
        return seq_model


if __name__ == "__main__":
    sm = ModelSeq().load_from_name("model_test", do_opt=True)
    inp = torch.rand([1, 128]).cuda()
    res = sm(inp)
    print(res)
