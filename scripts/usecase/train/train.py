import os
import hydra
import datetime
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.callbacks.CallbackTrainingTime import CallbackTrainingTime


config_name = "config_cdsvae_sprite"
# config_name = "config_cdsvae_dcalw"


@hydra.main(version_base=None, config_path="../../conf", config_name=config_name)
def run(config: DictConfig) -> None:
    if config.experiment.manual_seed is not None:
        seed_everything(config.experiment.manual_seed, True)
    print("------------------------------------")
    print("         seed : ", config.experiment.manual_seed)
    print("   datamodule : ", config.datamodule.name)
    print("         memo : ", config.memo)
    print("------------------------------------")

    data            = DataModuleFactory().create(**config.datamodule)
    lit_model_class = ModelFactory().create(config.model.name)
    lit_model       = lit_model_class(config, num_train=data.num_train)

    tb_logger = TensorBoardLogger(
        version  = '[{}]-[{}]-[dim_f={}]-[dim_z={}]-[{}epoch]-[{}]-{}'.format(
            config.model.name,
            config.datamodule.name,
            config.model.network.context_encoder.context_dim,
            config.model.network.motion_encoder.state_dim,
            config.trainer.max_epochs,
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
            config.memo,
        ),
        **config.logger
    )
    p = pathlib.Path(tb_logger.log_dir)
    p.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, tb_logger.log_dir + "/config.yaml")

    trainer = Trainer(
        logger    = tb_logger,
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(
                dirpath  = os.path.join(tb_logger.log_dir , "checkpoints"),
                filename = '{epoch}',
                **config.checkpoint,
            )
        ] + [CallbackTrainingTime()],
        **config.trainer
    )
    trainer.fit(model=lit_model, datamodule=data)


if __name__ == '__main__':
    run()