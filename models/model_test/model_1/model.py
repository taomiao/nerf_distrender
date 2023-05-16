import torch


def load_model():
    print("loading ...")
    m = ModelFront().cuda()
    return m


class ModelFront(torch.nn.Module):
    def __init__(self):
        super(ModelFront, self).__init__()
        self.linear = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        return self.relu(self.linear(inp))


import os
import torch._dynamo.config

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    # torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # m = ToyModel().cuda()
    m = ModelFront().cuda()
    m = torch.nn.parallel.DistributedDataParallel(m, static_graph=False)
    m.compile()

    inp = torch.rand([1, 128]).cuda()
    res = m(inp)
    print(res)
