import os

import torch
from torch.distributed.pipeline.sync.pipe import Pipe

from seq_model.seq_model import ModelSeq


class ModelPipe:
    def __init__(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
        self.piped_model = None

    def to_pipe(self, model):
        piped_model = Pipe(model)
        self.piped_model = piped_model
        return piped_model

