import torch

def load_model():
    print("loading ...")
    m = ModelFront("cuda")
    return m
class ModelFront(torch.nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(128, 64, device=device)

    def forward(self, inp):
        return self.fc1.forward(inp)