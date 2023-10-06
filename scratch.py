# %%
# %load_ext autoreload
# %autoreload 2
# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pytorch_lightning as pl
import torch
from xas.dataset import XASDataModule
from utils.src.optuna.dynamic_fc import PlDynamicFC


# %%
