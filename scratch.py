# %%
# %load_ext autoreload
# %autoreload 2
# %%

import pytorch_lightning as pl
import torch
from src.dataset import XASDataModule
from utils.src.optuna.dynamic_fc import PlDynamicFC


model = PlDynamicFC(widths=[800, 400], output_size=200)
trainer = pl.Trainer(max_epochs=100)
data_module = XASDataModule(dtype=torch.float32, num_workers=0)
trainer.fit(model, data_module)

# %%
