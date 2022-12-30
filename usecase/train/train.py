import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import hydra
from omegaconf import DictConfig, OmegaConf
from domain.train.Training import Training
from domain.callbacks.CallbackTrainingTime import CallbackTrainingTime



# config_name = "config"
config_name = "config_dsvae"

@hydra.main(version_base=None, config_path="../../conf", config_name=config_name)
def get_config(cfg: DictConfig) -> None:

    additionl_callbacks = [CallbackTrainingTime()]
    train = Training(cfg, additionl_callbacks)
    train.run()

get_config()

'''
tensorboard --logdir ./
'''