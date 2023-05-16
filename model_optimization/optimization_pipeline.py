import torch


class OptStage:

    def optimize(self, model: torch.nn.Module):
        pass


class InductorStage(OptStage):
    def optimize(self, model: torch.nn.Module):
        opt_model = torch.compile(model)
        return opt_model


class OptPipeline:
    def __init__(self):
        self.opt_stages = []
        self.opt_stages.append(InductorStage())

    def add_opt_stage(self, opt_stage: OptStage):
        self.opt_stages.append(opt_stage)

    def optimize(self, model: torch.nn.Module):
        for stage in self.opt_stages:
            model = stage.optimize(model)
        return model
