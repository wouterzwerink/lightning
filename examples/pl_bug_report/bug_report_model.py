import deepspeed
import torch
import torch.nn as nn

from pytorch_lightning.lite import LightningLite
from pytorch_lightning.strategies import DeepSpeedStrategy


class TheModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))



def run():
    class Lite(LightningLite):
        def run(self):
            model = TheModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer,
                dist_init_required=False,
            )


    Lite(strategy=DeepSpeedStrategy(stage=3, logging_batch_size_per_gpu=1), devices=2, accelerator="gpu").run()



if __name__ == "__main__":
    run()
