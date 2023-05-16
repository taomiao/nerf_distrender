from config import config
from model_distributed.distributed_model import DistModel
from model_optimization.opted_model import OptedModel
from model_pipeline.piped_model import ModelPipe
from seq_model.seq_model import ModelSeq


class ModelManager:
    def __init__(self):
        self.models = {

        }
        self.config = config

    def load_models(self):
        for m_name in self.config["models"]:
            m_conf = self.config["models"][m_name]
            if m_conf["load"] is True:
                model = ModelSeq().load_from_name(m_name, do_opt=m_conf["do_opt"])
                if m_conf["use_pipe"] is True:
                    model = ModelPipe().to_pipe(model)
                if m_conf["use_ddp"] is True:
                    model = DistModel().to_distributed(model)
                self.models[m_name] = model
        print(self.models)
