# %%
from pickle import dump, load
from p_tqdm import p_map
from src.data.raw_data import RAWData
from scripts.m3gnet_version_fix import fix_m3gnet_version
from src.data.multiway_paritioning import greedy_multiway_partition
from utils.src.plots.highlight_tick import highlight_tick
import multiprocessing
import re
from tqdm import tqdm
import os
import time
import pickle
import importlib
import src
from utils.src.misc import icecream
from utils.src.plots.highlight_tick import highlight_tick
from matplotlib import pyplot as plt
from src.data.feff_data_raw import RAWDataFEFF
import random
from copy import deepcopy
import scienceplots
import matplotlib as mpl
import numpy as np
from src.data.feff_data import FEFFData
from src.data.vasp_data_raw import RAWDataVASP
from itertools import combinations_with_replacement
from pprint import pprint
from typing import Tuple, TypedDict, Union
import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
import torch
import yaml
from scipy.signal import convolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scripts.pca_plots import linear_fit_of_pcas, plot_pcas
from scripts.plots_model_report import (
    heatmap_of_lines,
    plot_predictions,
    # plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_residue_quartiles,
)
from src.ckpt_predictions import get_optimal_fc_predictions
from src.plot.model_report import linear_model_predictions, model_report
from src.data.vasp_data import VASPData
from src.pl_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick

imports_1 = [
    "src",
    "utils.src.misc",
    "utils.src.plots.highlight_tick",
    "matplotlib.pyplot",
    "src.data.feff_data_raw",
    "scienceplots",
    "matplotlib",
    "src.data.feff_data",
    "src.data.vasp_data_raw",
    "scripts.pca_plots",
    "scripts.plots_model_report",
    "src.ckpt_predictions",
    "src.plot.model_report",
    "src.data.vasp_data",
    "src.pl_data",
    "utils.src.lightning.pl_module",
    "utils.src.optuna.dynamic_fc",
]


imports_2 = {
    "utils.src.misc": ["icecream"],
    "utils.src.plots.highlight_tick": ["highlight_tick"],
    "src.data.feff_data_raw": ["RAWDataFEFF"],
    "src.data.feff_data": ["FEFFData"],
    "src.data.vasp_data_raw": ["RAWDataVASP"],
    "src.data.vasp_data": ["VASPData"],
    "src.pl_data": ["XASData"],
    "utils.src.lightning.pl_module": ["PLModule"],
    "utils.src.optuna.dynamic_fc": ["PlDynamicFC"],
    "scripts.pca_plots": ["linear_fit_of_pcas", "plot_pcas"],
    "scripts.plots_model_report": [
        "heatmap_of_lines",
        "plot_predictions",
        "plot_residue_cv",
        "plot_residue_heatmap",
        "plot_residue_quartiles",
    ],
    "src.ckpt_predictions": ["get_optimal_fc_predictions"],
    "src.plot.model_report": ["linear_model_predictions", "model_report"],
}


def reimport_modules_and_functions():
    import importlib

    for module in imports_1:
        importlib.reload(importlib.import_module(module))
    for module, items in imports_2.items():
        reloaded_module = importlib.import_module(module)
        globals().update({item: getattr(reloaded_module, item) for item in items})


reimport_modules_and_functions()
# %%

compounds = ["Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Ti", "V"]
dir = os.path.join("FEFF-processed-data")
counts = [len(os.listdir(os.path.join(dir, c))) for c in compounds]
counts = sorted(list(zip(compounds, counts)), key=lambda x: x[1], reverse=True)


# %%


plt.style.use(["default", "science"])
plt.figure(figsize=(8, 6))
plt.bar(*zip(*counts), color="tab:blue")
plt.xlabel("Compound", fontsize=18)
plt.xticks(fontsize=14)
plt.ylabel("Number of Spectras", fontsize=18)
plt.yticks(fontsize=14)
for i, (c, n) in enumerate(counts):
    plt.text(i, n + 5, n, ha="center", va="bottom", fontsize=14)
plt.title("Number of FEFF Spectras for ML", fontsize=18)
plt.tight_layout()
plt.savefig("ml_feff_counts.pdf", dpi=300, bbox_inches="tight")

# %%
