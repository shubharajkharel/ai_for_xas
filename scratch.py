# %%
# %load_ext autoreload
# %autoreload 2
from compare_utils import compare_between_spectra
import numpy as np
from src.feff_data_transformations import FEFFDataModifier

# from scripts.plots_model_report import plot_residue_histogram
from src.raw_data_vasp import RAWDataVASP
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
from src.model_report import linear_model_predictions, model_report
from src.plot_vasp_transormations import VASPDataTransformationPlotter
from src.raw_data_feff import RAWDataFEFF
from src.vasp_data_transformations import VASPDataModifier
from src.xas_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick


# %%

compound = "Ti"
id = ("mp-390", "000_Ti")
vasp = RAWDataVASP(compound=compound)
vasp_spectra = VASPDataModifier(vasp.parameters[id])

feff = RAWDataFEFF(compound=compound)
feff_spectra = FEFFDataModifier(feff.parameters[id])

# %%


def align_to_vasp(source, target):
    # source = source - np.min(source)
    source = source / np.max(source)
    # source = source + np.min(target)
    source = source * np.max(target)
    return source


def shift_energy(spectra_1, spectra_2):
    shift, _ = compare_between_spectra(spectra_1, spectra_2)
    return shift


def filter_range(energy, spectra, target_energy_range):
    idx_min = np.where(energy > target_energy_range[0])[0][0]
    idx_max = np.where(energy < target_energy_range[-1])[0][-1]
    return energy[idx_min:idx_max], spectra[idx_min:idx_max]


vasp_spectra.energy, vasp_spectra.spectra = None, None
vasp_spectra.energy, vasp_spectra.spectra = vasp_spectra.truncate()
vasp_spectra.spectra -= np.min(vasp_spectra.spectra)
vasp_spectra.spectra = vasp_spectra.scale()
vasp_spectra.spectra = vasp_spectra.broaden(gamma=0.89 * 2)
vasp_spectra.energy = vasp_spectra.align()
vasp_spectra_transformed = np.array([vasp_spectra.energy, vasp_spectra.spectra]).T
vasp_label = "vasp_truc_minZero_scaled_broadened1.78_aligned"

e_vasp_max = vasp_spectra.energy[np.argmax(vasp_spectra.spectra)]

feff_spectra.energy, feff_spectra.spectra = None, None
feff_spectra.energy, feff_spectra.spectra = feff_spectra.truncate()
feff_spectra.spectra -= np.min(feff_spectra.spectra)
feff_spectra_transformed = np.array([feff_spectra.energy, feff_spectra.spectra]).T
feff_spectra.energy -= shift_energy(feff_spectra_transformed, vasp_spectra_transformed)
feff_spectra.energy, feff_spectra.spectra = filter_range(
    feff_spectra.energy, feff_spectra.spectra, vasp_spectra.energy
)
# feff_spectra.spectra = feff_spectra.broaden(gamma=0.89)
feff_spectra.spectra = align_to_vasp(feff_spectra.spectra, vasp_spectra.spectra)
feff_label = "feff_truc_minSetZero_alignedToVASP_scaledToVASP"

xs_mp_390 = np.load("dataset/xs-mp-390.npy")
xs_energy, xs_spectra = xs_mp_390[0], xs_mp_390[1]
xs_spectra -= np.min(xs_spectra)
xs_energy -= xs_energy[np.argmax(xs_spectra)] - e_vasp_max
xs_energy, xs_spectra = filter_range(xs_energy, xs_spectra, vasp_spectra.energy)
xs_spectra = VASPDataModifier.lorentz_broaden(xs_energy, xs_energy, xs_spectra, 0.89)
xs_spectra = align_to_vasp(xs_spectra, vasp_spectra.spectra)
xs_label = "xs_minSetZero_alignedToVASP_broadedned0.89_scaledToVASP"

anatase = np.loadtxt("dataset/anatase.txt")
anatase_energy = anatase[:, 0]
anatase_spectra = anatase[:, 1]
anatase_spectra -= np.min(anatase_spectra)
anatase_energy -= anatase_energy[np.argmax(anatase_spectra)] - e_vasp_max
anatase_energy, anatase_spectra = filter_range(
    anatase_energy, anatase_spectra, vasp_spectra.energy
)
anatase_spectra = align_to_vasp(anatase_spectra, vasp_spectra.spectra)
anatase_label = "anatase_minSetZero_alignedToVASP_scaledToVASP"

import scienceplots

import matplotlib as mpl

# mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(["vibrant", "no-latex"])
plt.figure(figsize=(8, 5))
plt.plot(vasp_spectra.energy, vasp_spectra.spectra, label=vasp_label)
plt.plot(feff_spectra.energy, feff_spectra.spectra, label=feff_label)
plt.plot(xs_energy, xs_spectra, label=xs_label)
plt.plot(anatase_energy, anatase_spectra, label=anatase_label)
plt.xlim(vasp_spectra.energy[0], vasp_spectra.energy[-1])
# legend outside the figure
plt.legend(fontsize=14, loc="lower left", bbox_to_anchor=(0, 1.01))
plt.savefig("alignments.pdf", bbox_inches="tight", dpi=300)
plt.show()


# %%
