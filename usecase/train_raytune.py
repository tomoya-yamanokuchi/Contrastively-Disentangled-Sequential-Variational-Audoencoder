import os
import time
import hydra
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from domain.raytune.SchedulerFactory import SchedulerFactory
from domain.raytune.SearchAlgorithmFactory import SearchAlgorithmFactory
from domain.notify.Notifying import Notifying
from Training import Training


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def get_config(cfg: DictConfig) -> None:
    time_start = time.time()

    recursive_defaultdict = lambda: defaultdict(recursive_defaultdict)
    config                = recursive_defaultdict()
    config["model"]["kld_weight"] = tune.choice([1e-2, 1e-3, 1e-4])
    config["model"]["conv_out_channels"] = tune.choice(
        [
            [32, 32, 32, 32],
            [32, 64, 64, 64],
            [32, 64, 128, 128],
            [32, 64, 128, 64]
        ]
    )

    metrics = {"loss": "loss"}

    train = Training(
        config              = cfg,
        additionl_callbacks = [TuneReportCallback(metrics, on="validation_end")]
    )

    scheduler  = SchedulerFactory().create(**cfg.raytune.scheduler)
    search_alg = SearchAlgorithmFactory().create(**cfg.raytune.search_algorithm, **cfg.raytune.common)
    analysis   = tune.run(train.run,
        config     = config,
        scheduler  = scheduler,
        search_alg = search_alg,
        **cfg.raytune.general,
        **cfg.raytune.common,
    )
    print("-------------------------------------------------------")
    print("Best hyperparameters found were: ", analysis.best_config)
    print("                      Best loss: ", analysis.best_result["loss"])
    print("                     Best trial: ", analysis.best_trial)
    print("-------------------------------------------------------")

    elapsed_time = time.time() - time_start
    notifying    = Notifying(path_web_hook_url=cfg.notify.path_web_hook_url)
    file_name    = __file__.split("/")[-1]
    notifying.notify_slack(file_name, elapsed_time)


get_config()