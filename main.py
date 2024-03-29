import logging
import sys
from config.defaults import hydra_config_is_valid

import hydra
import lightning
import lightning as pl
import optuna
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig
from rich import print

from src.data.ml_data import DataQuery, XASPlData, load_xas_ml_data
from utils.src.lightning.pl_module import PLModule


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    hydra_config_is_valid(cfg)
    data_module = instantiate(cfg.data_module)

    # # determine input and output dimensions from the data
    # data_sample = data_module.train_dataloader().dataset[0]
    # input_dim = data_sample[0].shape[0]
    # output_dim = data_sample[1].shape[0]
    # widths = [input_dim] + cfg.model.widths + [output_dim]
    # torch_model = instantiate(cfg.model, widths=widths)

    torch_model = instantiate(cfg.model)

    pl_model = PLModule(torch_model)

    # TODO: Fix this error: it is sending removing the batch dimension for some reason
    # model.example_input_array = data_module.train_dataloader().dataset[0][0].to("mps")

    trainer = instantiate(cfg.trainer)

    # TODO: Fix this error: this is getting trainer initilization stuck
    # trainer.callbacks = instantiate(cfg.callbacks).values()

    trainer.fit(pl_model, data_module)
    trainer.test(pl_model, datamodule=data_module)


class Optimizer:
    def __init__(self, hydra_configs):
        self.cfg = hydra_configs

    def optimize(self, trial):

        # depth = trial.suggest_int("depth", 1, 3)
        depth = trial.suggest_int(
            "depth",
            self.cfg.optuna.params.min_depth,
            self.cfg.optuna.params.max_depth,
        )

        hidden_widths = [
            trial.suggest_int(
                f"width_{i}",
                self.cfg.optuna.params.min_width,
                self.cfg.optuna.params.max_width,
                self.cfg.optuna.params.step_width,
            )
            for i in range(depth)
        ]

        # learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        # self.cfg["model"]["learning_rate"] = learning_rate
        # model = instantiate(self.cfg.model)

        # batch_exponent = trial.suggest_int("batch", 6, 9, 1)
        # batch_size = 2**batch_exponent
        # self.cfg["data_module"]["batch_size"] = batch_size

        # widths = [64] + widths + [141]

        data_module = instantiate(self.cfg.data_module)

        data_sample = data_module.train_dataloader().dataset[0]
        input_width = data_sample[0].shape[0]
        output_width = data_sample[1].shape[0]

        widths = [input_width] + hidden_widths + [output_width]

        torch_model = instantiate(self.cfg.model, widths=widths)
        pl_model = PLModule(torch_model)

        # TODO: Fix this error: it is sending removing the batch dimension for some reason
        # model.example_input_array = (
        #     data_module.train_dataloader().dataset[0][0].to("mps")
        # )  # needed for loggers

        # traint and test
        trainer = instantiate(self.cfg.trainer)

        # TODO: Fix this error: this is getting trainer initilization stuck
        # trainer.callbacks = []  # removes progressbar
        # for cb_name, cb_cfg in self.cfg.callbacks.items():
        #     cb_instance = instantiate(cb_cfg)
        #     trainer.callbacks.append(cb_instance)

        # TODO: make this load from cfg after fixing the error above
        trainer.callbacks.extend(
            [
                lightning.pytorch.callbacks.lr_finder.LearningRateFinder(),
                lightning.pytorch.callbacks.early_stopping.EarlyStopping(
                    monitor="val_loss", patience=10, mode="min", verbose=False
                ),
            ]
        )

        # trainer.callbacks.append(
        #     PyTorchLightningPruningCallback(trial, monitor="val_loss")
        # )

        trainer.fit(pl_model, data_module)

        return trainer.callback_metrics["val_loss"]


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def run_optmization(cfg: DictConfig):
    hydra_config_is_valid(cfg)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=cfg.optuna.storage,
        # storage="sqlite:///{}.db".format("test"),
        load_if_exists=True,
    )
    optimizer_class = Optimizer(hydra_configs=cfg)
    study.optimize(optimizer_class.optimize, n_trials=cfg.optuna.n_trials)
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)
    # print("Trials: ", study.trials)


if __name__ == "__main__":
    # run_optmization()
    main()
