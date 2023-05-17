import torch
from tensoRF import TensorVMSplit
from opt import config_parser
def load_model():
    ckpt_path = "/home/PJLAB/taomiao/PycharmProjects/distrender/models/model_leftsmall/model_1/2_leftsmall_ds10.th"
    conf_path = "/home/PJLAB/taomiao/PycharmProjects/distrender/models/model_leftsmall/model_1/config.txt"
    device = "cuda"
    args = config_parser(cmd=f"--config {conf_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device, 'args': args})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)
    # tensorf.half()
    tensorf.eval()
    return tensorf