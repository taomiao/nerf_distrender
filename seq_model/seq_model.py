import os
import sys

import torch

from config import config
from seq_model.model import Model
from model_optimization.opted_model import OptedModel

models_repo = config["models_repo"]


class SequentialMultiInput(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ModelSeq:
    def __init__(self):
        sys.path.append(models_repo)
        self.models = []
        self.seq_model = None

    def load_from_name(self, name, do_opt=False) -> torch.nn.Sequential:
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
        seq_model = SequentialMultiInput(*self.models)
        self.seq_model = seq_model
        return seq_model
