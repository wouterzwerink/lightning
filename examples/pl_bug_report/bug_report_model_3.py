from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset


class TestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        loss = self.step(batch[0])
        opt = self.optimizers()
        # opt.zero_grad()
        print("debug", batch_idx, opt, self.layer.weight.grad)
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        return {"loss": loss}

def run():
    model = TestModel()
    train_batches = 2
    val_batches = 2
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        enable_progress_bar=False,
        enable_model_summary=False,
        track_grad_norm=1,
        accelerator="gpu", devices=2, precision=16, strategy="deepspeed"
    )
    trainer.fit(model)


if __name__ == "__main__":
    run()