import os

import torch
from torch.nn.parallel import DistributedDataParallel

from model_pipeline.piped_model import ModelPipe
from seq_model.seq_model import ModelSeq


class DistModel:
    def __init__(self):
        self.ddp_model = None
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        # torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    def to_distributed(self, model):
        ddp_model = DistributedDataParallel(model)
        self.ddp_model = ddp_model
        return ddp_model
