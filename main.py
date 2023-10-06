import optuna
import torch
import pytorch_lightning as pl
import optuna
import sqlite3
import os
from pytorch_lightning.loggers import TensorBoardLogger


# from models.baseline_models import StaticBaseline
import hydra
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.src.optuna.dynamic_fc import PlDynamicFC


class DynamicFCOptunaObjective:
    def __init__(
        self,
        hydra_cfg,
        input_size=64,
        min_depth=1,
        max_depth=1,
        min_width=10,
        max_width=10,
    ):
        self.input_size = input_size
        # self.data_module = instantiate(
        #     hydra_cfg.data_module
        # )  # this might assign same data loaders to all n_jobs
        self.hydra_cfg = hydra_cfg
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, trial):
        # hyperparameters
        depth = trial.suggest_int(
            "depth",
            self.hydra_cfg.optimization.min_depth,
            self.hydra_cfg.optimization.max_depth,
        )
        widths = [self.hydra_cfg.model.input_size]

        widths += [
            trial.suggest_int(
                f"width_{i}",
                self.hydra_cfg.optimization.min_width,
                self.hydra_cfg.optimization.max_width,
            )
            for i in range(depth)
        ]

        # widths += [
        #     trial.suggest_int(f"width_{i}", self.min_width, self.max_width)
        #     for i in range(depth)
        # ]

        self.data_module = instantiate(self.hydra_cfg.data_module)

        loss = instantiate(self.hydra_cfg.model.loss)
        model = PlDynamicFC(
            widths, loss=loss, output_size=self.hydra_cfg.model.output_size
        )
        data_module = instantiate(self.hydra_cfg.data_module)
        # logger = TensorBoardLogger(save_dir="", version="trial_" + str(trial.number))
        # trainer = instantiate(self.hydra_cfg.trainer, logger=logger)
        trainer = instantiate(self.hydra_cfg.trainer)
        trainer.fit(model, data_module)

        # optmization metrics
        # param_count = TorchAdapter(model).count_parameters()
        # total_bits = total_torch_bits(model)
        val_loss = trainer.callback_metrics["val_loss"].item()
        return val_loss


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def setup_distributed(cfg: DictConfig):
    # Create SQLite database if it doesn't exist
    sql_file_name = cfg.study.study_name + ".db"
    if os.path.exists(cfg.study.storage):
        print("Database exits, either remove it or back it up")
        exit(1)
    else:
        print("Creating database")
        conn = sqlite3.connect(sql_file_name)
        conn.close()


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    study = (
        hydra.utils.call(cfg.study)
        if not cfg.optimization.run_distributed
        else optuna.load_study(
            study_name=cfg.study.study_name, storage=cfg.study.storage
        )
    )

    objective = DynamicFCOptunaObjective(hydra_cfg=cfg)

    study.optimize(
        objective,
        n_trials=cfg.optimization.n_trials,
        timeout=cfg.optimization.timeout,
        n_jobs=cfg.optimization.n_jobs,
    )  # n_jobs will for each process if calling from command line

    # ax = optuna.visualization.matplotlib.plot_contour(study)


# @hydra.main(version_base=None, config_path="cfg", config_name="config")
# def profile_max_baseline(cfg: DictConfig):
#     logger = TensorBoardLogger(save_dir="", version="baseline_max")
#     loss = instantiate(cfg.model.loss)
#     Trainer(devices=1, num_nodes=1, logger=logger).test(
#         PLModule(StaticBaseline(), loss=loss), instantiate(cfg.data_module)
#     )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    # profile_max_baseline()
    # setup_distributed()
    main()

    # compound_name = "Cu-O"
    # data_module = XASDataModule(dtype=torch.float32, num_workers=0, compound=compound_name)
    # model = PlDynamicFC(widths=[64, 400], output_size=200)
    # trainer = pl.Trainer(max_epochs=10)
    # trainer.fit(model, data_module)

    # to save sampler
    # not sure why
    # with open("optuna_sampler", "wb") as fout:
    #     pickle.dump(study.sampler, fout)

    # You are usingssh a CUDA device ('NVIDIA GeForce RTX 3080 Ti Laptop GPU') that
    # has Tensor Cores. To properly utilize them, you should set
    # `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off
    # precision for performance. For more details, read
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#distributed
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    # n_jobs allows parallelization using threading and may suffer from Python’s
    # GIL. It is recommended to use process-based parallelization if func is CPU
    # bound.
    # use load and run setup before it
