from deepspeed import DeepSpeedEngine

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset
from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin

config = {
    "activation_checkpointing": {
        "contiguous_memory_optimization": False,
        "cpu_checkpointing": False,
        "partition_activations": False,
        "synchronize_checkpoint_boundary": False,
    },
    "aio": {"block_size": 1048576, "overlap_events": True, "queue_depth": 8, "single_submit": False, "thread_count": 1},
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0.0,
    "train_micro_batch_size_per_gpu": 1,
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "allgather_bucket_size": 200000000,
        "allgather_partitions": True,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_bucket_size": 200000000,
        "reduce_scatter": True,
        "stage": 2,
        "sub_group_size": 1000000000000,
    },
}


class TestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        loss = self.step(batch[0])
        opt = self.optimizers()
        assert isinstance(self.trainer.precision_plugin, DeepSpeedPrecisionPlugin)
        assert isinstance(self.trainer.model, DeepSpeedEngine)
        # self.manual_backward(loss)
        # opt.zero_grad()
        self.trainer.model.backward(loss)
        # self.trainer.precision_plugin.backward(self, loss, None)
        self.trainer.model.step()
        # opt.step()
        # opt.zero_grad()
        return {"loss": loss}


def run():
    model = TestModel()
    train_batches = 2
    val_batches = 2
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        enable_progress_bar=True,
        enable_model_summary=False,
        track_grad_norm=1,
        accelerator="gpu",
        devices=1,
        strategy="deepspeed",
    )
    trainer.fit(model)


if __name__ == "__main__":
    run()
