import torch
from tensoRF_stage2 import TensorVMSplitStage2
from opt import config_parser
def load_model():
    ckpt_path = "/home/PJLAB/taomiao/PycharmProjects/distrender/models/model_leftsmall_multi_stages/model_2/2_leftsmall_ds10.th"
    conf_path = "/home/PJLAB/taomiao/PycharmProjects/distrender/models/model_leftsmall_multi_stages/model_2/config.txt"
    device = "cpu"
    args = config_parser(cmd=f"--config {conf_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device, 'args': args})
    tensorf = TensorVMSplitStage2(**kwargs)
    tensorf.load(ckpt)
    # tensorf.half()
    tensorf.eval()
    return tensorf