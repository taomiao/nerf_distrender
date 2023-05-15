import torch

def load_model():
    print("loading ...")
    m = ModelEnd("cuda")
    return m
class ModelEnd(torch.nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc2 = torch.nn.Linear(64, 2, device=device)

    def forward(self, inp):
        return self.fc2.forward(inp)