import torch
from torch.distributed.pipeline.sync.pipe import Pipe
from model_pipeline.seq_model import ModelSeq
import os
class ModelPipe:
    def __init__(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
        self.piped_model = None

    def to_pipe(self, model: ModelSeq):
        piped_model = Pipe(model.seq_model)
        self.piped_model = piped_model
        return piped_model

if __name__=="__main__":
    sm = ModelSeq()
    sm.load_from_name("model_test", do_opt=True)
    pm = ModelPipe().to_pipe(sm)
    inp = torch.rand([1, 128]).cuda()
    res = pm(inp)
    print(res.to_here())