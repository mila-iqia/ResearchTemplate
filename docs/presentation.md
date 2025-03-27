
===
# Project Structure

```
Research Project Code Structure
├── Configs (Hydra configs)
│
├── Datamodule (data preprocessing / dataloading)
│   ├── Dataset
│   └── Transforms
│
├── Algorithm (learning algorithm)
│   ├── Network
│   └── Optimizer
│
└── Trainer (training loop, distributed training, etc.)
    ├── Callbacks
    ├── Logging
    └── Checkpointing
```



# Datamodule

```python
class LightningDataModule:
    def prepare_data(self):
        """Dataset preparation (e.g. download / preprocessing). Called only once."""
    def setup(self, stage: str):
        """Called at the start of train/val/test on each replica."""
    def train_dataloader(self) -> DataLoader:
        ...
    def val_dataloader(self) -> DataLoader:
        ...
    def test_dataloader(self) -> DataLoader:
        ...
```

<!-- end_slide -->

# Trainer

```python
class Trainer:
    callbacks: list[Callback]
    loggers: list[Logger]
    def fit(self,
            module: LightningModule,
            datamodule: LightningDataModule | None, ...):
        ...
    def evaluate(...):
        ...
    def test(...):
```

<!-- end_slide -->

# Trainer

```python
class LightningModule:
    # Required:
    def training_step(self, batch, batch_idx: int) -> Tensor:
        ...
    def configure_optimizers(self) -> Optimizer:
        ...
    # Optional, recommended:
    def validation_step(self, batch, batch_idx: int) -> Tensor:
        ...
    def test_step(self, batch, batch_idx: int) -> Tensor:
        ...
```

<!-- end_slide -->

# How are they used?

```python
from hydra.utils import instantiate

@hydra.main
def main(config)
    datamodule = instantiate(config.datamodule)
    algorithm = instantiate(config.algorithm, datamodule=datamodule)
    trainer = instantiate(config.trainer)
    trainer.fit(algorithm, datamodule=datamodule)
    metrics = trainer.evaluate(algorithm, datamodule=datamodule)
    return metrics
```


<!-- end_slide -->

# Demo commands:

```bash +exec
python project/main.py experiment=cluster_sweep_example cluster=current
```
