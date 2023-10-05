import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


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
        train_dataset = self.load_dataset(compound=compound, split=split, task="train")
        val_dataset = self.load_dataset(compound=compound, split=split, task="val")
        test_dataset = self.load_dataset(compound=compound, split=split, task="test")
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **pl_data_module_kwargs,
        )

    def load_dataset(
        self,
        data_dir: Union[str, Path] = "dataset/ML",
        compound: Union[str, Path] = "Cu-O",
        split: Literal["material-splits", "random-splits"] = "material-splits",
        task: Literal["train", "val", "test"] = "train",
    ):
        data_compound_dir = os.path.join(data_dir, compound, split, "data")
        X_data = np.load(os.path.join(data_compound_dir, f"X_{task}.npy"))
        y_data = np.load(os.path.join(data_compound_dir, f"y_{task}.npy"))
        if X_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"X_{task}.npy and y_{task}.npy have different lengths \
                for compound {compound} and split_type {split}"
            )
        return TensorDataset(
            torch.from_numpy(X_data).to(dtype=self.dtype),
            torch.from_numpy(y_data).to(dtype=self.dtype)
            # torch.from_numpy(X_data).to(device=self.device),
            # torch.from_numpy(y_data).to(device=self.device),
        )


if __name__ == "__main__":
    data_module = XASDataModule()
    print(data_module.train_dataset[0][0].shape)
    print(data_module.train_dataset[0][1].shape)
