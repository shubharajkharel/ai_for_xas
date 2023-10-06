import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset
from xas.data_heatmaps import plot_heatmap


from utils.src.lightning.pl_data_module import PlDataModule


class XASDataModule(PlDataModule):
    def __init__(
        self,
        compound: Union[str, Path] = "Cu-O",
        split: Literal["material-splits", "random-splits"] = "material-splits",
        task: Literal["train", "val", "test"] = "train",
        dtype: torch.dtype = torch.float32,
        **pl_data_module_kwargs,
    ):
        self.compound = compound
        self.split = split
        self.task = task
        self.dtype = dtype
        train_dataset = self.np_data_to_dataset(
            *self.load_data(compound=compound, split=split, task="train", dtype=dtype)
        )
        val_dataset = self.np_data_to_dataset(
            *self.load_data(compound=compound, split=split, task="val", dtype=dtype)
        )
        test_dataset = self.np_data_to_dataset(
            *self.load_data(compound=compound, split=split, task="test", dtype=dtype)
        )
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **pl_data_module_kwargs,
        )

    def np_data_to_dataset(self, x, y, device="cpu"):
        return TensorDataset(
            torch.from_numpy(x).to(dtype=self.dtype),
            torch.from_numpy(y).to(dtype=self.dtype),
        )

    @classmethod
    def load_data(
        self,
        data_dir: Union[str, Path] = "dataset/ML",
        compound: Union[str, Path] = "Cu-O",
        split: Literal["material-splits", "random-splits"] = "material-splits",
        task: Literal["train", "val", "test"] = "train",
        dtype: torch.dtype = torch.float32,
    ):
        data_compound_dir = os.path.join(data_dir, compound, split, "data")
        x = np.load(os.path.join(data_compound_dir, f"X_{task}.npy"))
        y = np.load(os.path.join(data_compound_dir, f"y_{task}.npy"))
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_{task}.npy and y_{task}.npy have different lengths \
                for compound {compound} and split_type {split}"
            )
        return x, y


# def plot_spectral_heatmap(compound, task, data, min_val, max_val, title_prefix):
#     plot_heatmap(
#         y_data=data,
#         title=f"{title_prefix} for {compound} in {task}",
#         save=True,
#         yticks_values=np.linspace(min_val, max_val, 10),
#     )


if __name__ == "__main__":
    pass
    # data_module = XASDataModule()
    # print(data_module.train_dataset[0][0].shape)
    # print(data_module.train_dataset[0][1].shape)

    # compounds = ["Cu-O", "Ti-O"]
    # tasks = ["train", "test"]
    # for idx, title in [(1, "Target distribution"), (0, "Input distribution")]:
    #     for compound in compounds:
    #         for task in tasks:
    #             data = XASDataModule.load_data(compound=compound, task=task)[idx]
    #             data = (data - data.min()) / (data.max() - data.min())
    #             plot_spectral_heatmap(
    #                 compound, task, data, data.min(), data.max(), title
    #             )
