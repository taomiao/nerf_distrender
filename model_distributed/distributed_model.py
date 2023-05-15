import torch
from torch.nn.parallel import DistributedDataParallel

from model_pipeline.piped_model import ModelPipe
from model_pipeline.seq_model import ModelSeq
import os

class DistModel:
    def __init__(self):
        self.ddp_model = None
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        #torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    def to_distributed(self, model):
        ddp_model = DistributedDataParallel(model)
        self.ddp_model = ddp_model
        return ddp_model


if __name__=="__main__":
    sm = ModelSeq()
    sm.load_from_name("model_test")
    pm = ModelPipe().to_pipe(sm)
    dm = DistModel().to_distributed(pm)
    inp = torch.rand([1, 128]).cuda()
    res = pm(inp)
    print(res)