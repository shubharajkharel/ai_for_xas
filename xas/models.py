from xas.dataset import XASDataModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
import pytorch_lightning as pl
import torch


if __name__ == "__main__":
    compound_name = "Cu-O"
    data_module = XASDataModule(
        dtype=torch.float32, num_workers=0, compound=compound_name
    )
    model = PlDynamicFC(widths=[64, 400], output_size=200)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
