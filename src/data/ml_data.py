from sklearn.preprocessing import StandardScaler
from functools import cached_property
import pickle
from typing import Union
import os
import itertools
from dataclasses import dataclass
from typing import Any, List, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from config.defaults import cfg

from src.data.material_split import MaterialSplitter
from utils.src.lightning.pl_data_module import PlDataModule
from dgl import DGLHeteroGraph


@dataclass
class DataSplit:
    X: Union[np.ndarray, DGLHeteroGraph]
    y: np.ndarray

    def __post_init__(self):
        if not len(self.X) == len(self.y):
            raise ValueError("X and y must have same length")
        if self.X.dtype != self.y.dtype and not isinstance(self.X[0], DGLHeteroGraph):
            raise ValueError("X and y ndarray must have same dtype")

    @property
    def tensors(self):
        x_is_graph = isinstance(self.X[0], DGLHeteroGraph)
        X = self.X if x_is_graph else torch.tensor(self.X)
        y = torch.tensor(self.y)
        return X, y


@dataclass
class MLSplits:
    train: DataSplit
    val: DataSplit
    test: DataSplit

    @property
    def tensors(self):
        if isinstance(self.train.X, DGLHeteroGraph):
            raise NotImplementedError("DGLHeteroGraph->Tensor not implemented.")
        out_tensor = (self.train.tensors, self.val.tensors, self.test.tensors)
        flatten = itertools.chain.from_iterable
        return tuple(flatten(out_tensor))


@dataclass
class DataQuery:
    compound: str
    simulation_type: Literal["FEFF", "VASP"]


class FeatureProcessor:
    def __init__(self, query: DataQuery, data_splits: MLSplits = None):
        self.query = query
        # none option set to access saved pca and scaler from cache
        self.data_splits = data_splits

    @cached_property
    def splits(self):
        return MLSplits(
            train=self._reduce_feature_dims(self.data_splits.train),
            val=self._reduce_feature_dims(self.data_splits.val),
            test=self._reduce_feature_dims(self.data_splits.test),
        )

    def _reduce_feature_dims(self, data_splits: DataSplit):
        return DataSplit(
            self.pca.transform(self.scaler.transform(data_splits.X)), data_splits.y
        )

    @cached_property
    def scaler(self):
        # load from cache if cache exists
        scaler_cache_path = cfg.paths.cache.scaler.format(**self.query.__dict__)
        if os.path.exists(scaler_cache_path):  # load from cache
            with open(scaler_cache_path, "rb") as f:
                return pickle.load(f)
        # else fit scaler and save to cache
        if self.data_splits is None:
            raise ValueError("data_splits is None. Cannot fit scaler.")
        scaler = StandardScaler().fit(self.data_splits.train.X)
        os.makedirs(os.path.dirname(scaler_cache_path), exist_ok=True)
        with open(scaler_cache_path, "wb") as f:
            pickle.dump(scaler, f)
        return scaler

    @cached_property
    def pca(self):
        # load from cache if cache exists
        pca_cache_path = cfg.paths.cache.pca.format(**self.query.__dict__)
        if os.path.exists(pca_cache_path):
            with open(pca_cache_path, "rb") as f:
                return pickle.load(f)
        # else fit pca and save to cache
        if self.data_splits is None:
            raise ValueError("data_splits is None. Cannot fit pca.")
        pca = PCA(n_components=cfg.dscribe.pca.n_components)
        pca.fit(self.data_splits.train.X)
        os.makedirs(os.path.dirname(pca_cache_path), exist_ok=True)
        with open(pca_cache_path, "wb") as f:
            pickle.dump(pca, f)
        return pca

    def _test_if_pca_matches_config(self):
        # expected number of components or the explained variance
        expected_pca_param = cfg.dscribe.pca.n_components_
        if not self.pca.n_components == expected_pca_param:
            msg = "PCA components mismatch: "
            msg += f"{self.pca.n_components_} != {expected_pca_param} for {self.query}"
            raise ValueError(msg)


def load_xas_ml_data(
    query: DataQuery,
    split_fractions: Union[List[float], None] = None,
    reduce_dims: Union[bool, None] = None,
    normalize: bool = True,
    for_m3gnet: bool = False,
) -> MLSplits:
    """Loads data and does material splitting."""

    if query.compound == "ALL":  # TODO: hacky
        return load_all_data(query.simulation_type, split_fractions=split_fractions)

    file_path = OmegaConf.load("./config/paths.yaml").paths.ml_data
    file_path = file_path.format(**query.__dict__)
    data_all = np.load(file_path, allow_pickle=True)

    # greedy multiway partitioning
    idSite = list(zip(data_all["ids"], data_all["sites"]))
    train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
        idSite=idSite,
        target_fractions=split_fractions or cfg.data_module.split_fractions,
    )

    def to_split(ids):
        material_ids = ids[:, 0]
        sites = ids[:, 1]
        id_match = np.isin(data_all["ids"], material_ids)
        site_match = np.isin(data_all["sites"], sites)
        filter = np.where(id_match & site_match)[0]
        X = data_all["features"][filter]
        x_not_graph = "GRAPH" not in query.simulation_type
        if x_not_graph:
            X = X.astype(np.float32)
        y = data_all["spectras"][filter].astype(np.float32)
        return DataSplit(X, y)

    splits = MLSplits(
        train=to_split(train_idSites),
        val=to_split(val_idSites),
        test=to_split(test_idSites),
    )

    if reduce_dims is None:
        reduce_dims = query.simulation_type in cfg.dscribe.features
    out = FeatureProcessor(query, splits).splits if reduce_dims else splits

    if normalize:  # TODO: use standard scaler or stg
        if not isinstance(out.train.X[0], DGLHeteroGraph):
            out.train.X *= 1000
            out.val.X *= 1000
            out.test.X *= 1000
        out.train.y *= 1000
        out.val.y *= 1000
        out.test.y *= 1000

    return out


class XASPlData(PlDataModule):
    def __init__(
        self,
        query: DataQuery,
        dtype: torch.dtype = torch.float,
        split_fractions: Union[List[float], None] = None,
        **data_loader_kwargs,
    ):
        def dataset(split: DataSplit):
            X, y = split.tensors
            X = X if isinstance(X[0], DGLHeteroGraph) else X.type(dtype)
            y = y.type(dtype)
            return TensorDataset(X, y)
            # return TensorDataset(X.type(dtype), y.type(dtype))

        split_fractions = split_fractions or cfg.data_module.split_fractions

        ml_split = load_xas_ml_data(query, split_fractions=split_fractions)
        super().__init__(
            train_dataset=dataset(ml_split.train),
            val_dataset=dataset(ml_split.val),
            test_dataset=dataset(ml_split.test),
            **data_loader_kwargs,
        )


def load_all_data(
    sim_type="FEFF",
    same_size=False,  # TODO:
    split_fractions=None,
    seed=42,
    return_compound_name=False,
):
    def ml_split_size(ml_split):
        return len(ml_split.train.X) + len(ml_split.val.X) + len(ml_split.test.X)

    data_all = load_xas_ml_data(DataQuery(cfg.compounds[0], sim_type))
    for c in cfg.compounds[1:]:
        data = load_xas_ml_data(DataQuery(c, sim_type))
        data_all = MLSplits(
            train=DataSplit(
                X=np.concatenate([data_all.train.X, data.train.X]),
                y=np.concatenate([data_all.train.y, data.train.y]),
            ),
            val=DataSplit(
                X=np.concatenate([data_all.val.X, data.val.X]),
                y=np.concatenate([data_all.val.y, data.val.y]),
            ),
            test=DataSplit(
                X=np.concatenate([data_all.test.X, data.test.X]),
                y=np.concatenate([data_all.test.y, data.test.y]),
            ),
        )
    if same_size:
        min_size = min(
            [
                ml_split_size(load_xas_ml_data(DataQuery(c, sim_type)))
                for c in cfg.compounds
            ]
        )
        split_fractions = split_fractions or cfg.data_module.split_fractions
        train_size = int(min_size * split_fractions[0])
        val_size = int(min_size * split_fractions[1])
        test_size = min_size - train_size - val_size
        data_all = MLSplits(
            train=DataSplit(
                X=data_all.train.X[:train_size], y=data_all.train.y[:train_size]
            ),
            val=DataSplit(X=data_all.val.X[:val_size], y=data_all.val.y[:val_size]),
            test=DataSplit(
                X=data_all.test.X[:test_size], y=data_all.test.y[:test_size]
            ),
        )

    data_per_compound = {
        c: load_xas_ml_data(DataQuery(c, sim_type), split_fractions=split_fractions)
        for c in cfg.compounds
    }
    train_compounds, val_compounds, test_compounds = [], [], []
    for compound, data in data_per_compound.items():
        train_size = len(data.train.X)
        val_size = len(data.val.X)
        test_size = len(data.test.X)
        train_compounds.extend([compound] * train_size)
        val_compounds.extend([compound] * val_size)
        test_compounds.extend([compound] * test_size)
    train_compounds = np.array(train_compounds)
    val_compounds = np.array(val_compounds)
    test_compounds = np.array(test_compounds)

    np.random.seed(seed)
    idx = np.random.permutation(len(data_all.train.X))
    data_all.train.X = data_all.train.X[idx]
    data_all.train.y = data_all.train.y[idx]
    train_compounds = train_compounds[idx]
    idx = np.random.permutation(len(data_all.val.X))
    data_all.val.X = data_all.val.X[idx]
    data_all.val.y = data_all.val.y[idx]
    val_compounds = val_compounds[idx]
    idx = np.random.permutation(len(data_all.test.X))
    data_all.test.X = data_all.test.X[idx]
    data_all.test.y = data_all.test.y[idx]
    test_compounds = test_compounds[idx]

    compound_list = train_compounds, val_compounds, test_compounds
    return (data_all, compound_list) if return_compound_name else data_all


if __name__ == "__main__":

    from src.models.xas_m3gnet import XASM3GNet
    from utils.src.lightning.pl_module import PLModule
    from lightning import Trainer

    model = XASM3GNet()
    pl_model = PLModule(model)
    trainer = Trainer()
    data_module = XASPlData(DataQuery("Cu", "FEFF_GRAPH"))
    trainer.fit(pl_model, data_module)


"""     # =============================================================================
    # tests if pca and scaler are cached
    # =============================================================================
    from p_tqdm import p_map

    # load_xas_ml_data(DataQuery("Cu", "SOAP"))
    load_xas_ml_data(DataQuery("Cu", "FEFF_GRAPH"))

    # # should cache pca and scaler
    # p_map(
    #     lambda q: load_xas_ml_data(q),
    #     [DataQuery(c, "SOAP") for c in cfg.compounds[-1]],
    #     num_cpus=1,
    # )

    # for compound in cfg.compounds:
    #     query = DataQuery(compound=compound, simulation_type="SOAP")
    #     # caching pca
    #     ml_split = load_xas_ml_data(query)
    #     pca = FeatureProcessor(query).pca  # should load from cache
    #     print(f"PCA components: {pca.n_components_} for {query}")

    # =============================================================================
    print("dummy")

    # pl_data = XASPlData(query=DataQuery(compound="Cu", simulation_type="FEFF"))
    # def print_fractions(xas_data):
    #     dataset = [xas_data.train_dataset, xas_data.val_dataset, xas_data.test_dataset]
    #     sizes = np.array([len(x) for x in dataset])
    #     sizes = sizes / sizes.sum()
    #     print(sizes)
    # print_fractions(pl_data)  # [0.8 0.1 0.1]
 """
