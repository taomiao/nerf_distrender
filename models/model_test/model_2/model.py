import torch


def load_model():
    print("loading ...")
    m = ModelEnd().cuda()
    return m


class ModelEnd(torch.nn.Module):
    def __init__(self):
        super(ModelEnd, self).__init__()
        self.fc2 = torch.nn.Linear(192, 2)

    def forward(self, inp1, inp2, inp3):
        inp = torch.cat([inp1, inp2, inp3])
        return self.fc2(inp)
