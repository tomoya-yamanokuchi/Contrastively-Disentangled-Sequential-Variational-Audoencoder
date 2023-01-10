
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# ------------------------------------
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.model.ModelFactory import ModelFactory
from domain.test.TestModel import TestModel
from domain.test import metric_for_gaussian as metric
from custom.utility.to_numpy import to_numpy


class TestDClaw:
    def load_model(self, group, model):
        log_dir = "/hdd_mount/logs_cdsvae/{}/{}".format(group, model)
        test    = TestModel(config_dir=log_dir, checkpoints="last.ckpt")
        self.model, self.config_model = test.load_model()


    def load_evaluator(self, group, model):
        log_dir = "/hdd_mount/logs_cdsvae/{}/{}".format(group, model)
        test    = TestModel(config_dir=log_dir, checkpoints="last.ckpt")
        self.evaluator, self.config_eval = test.load_model()


    def load_evaluation_dataset(self):
        # << training dataset for calculating marginal predictive distribution >>
        datamodule = DataModuleFactory().create(**self.config_eval.datamodule)
        datamodule.setup(stage="fit")
        self.train_dataloader = datamodule.train_for_test_dataloader()
        # datamodule.setup(stage="test")
        # self.train_dataloader = datamodule.test_dataloader() # デバッグ終わったらtrainになおして消して！！！

        # << evaluation dataset >>
        datamodule = DataModuleFactory().create(**self.config_eval.datamodule)
        datamodule.setup(stage="test")
        self.test_dataloader = datamodule.test_dataloader()


    def get_evaluator_predictive_distribution(self):
        mean   = []
        var    = []
        y_true = []
        for index, data in self.train_dataloader:
            x = data['input']  # image
            y = data['output'] # state

            # x_sampel = self.model.forward(x)
            x_sampel     = x
            results_dict = self.evaluator.forward(x_sampel)

            # convert from tensor to numpy
            y             = to_numpy(y)
            mean_ensemble = to_numpy(results_dict["mean"])
            var_ensemble  = to_numpy(results_dict["var"])

            # calculate predictive distribution
            mean_pred = mean_ensemble.mean(axis=0)
            var_pred  = (var_ensemble + mean_ensemble**2).mean(axis=0) - mean_pred**2

            mean.append(mean_pred)
            var.append(var_pred)
            y_true.append(y)

        mean = np.concatenate(mean, axis=0)
        var  = np.concatenate(var, axis=0)
        # y_true = np.concatenate(y_true, axis=0)
        return (mean, var)


    def get_evaluator_marginal_predictive_distribution(self, mean, var):
        return (mean.mean(axis=0), var.mean(axis=0))



if __name__ == '__main__':

    model_model = "[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[300epoch]-[20230105142322]-melco_dclaw_config_cdsvae_dclaw"
    group_model = "cdsvae_dclaw"

    model_eval  = "[regressor_dclaw]-[action_norm_valve]-[dim_out=8]-[dim_fc_hidden=256]-[100epoch]-[num_batch=128]-[beta=0.5]-[20230109222005]-[melco]-"
    group_eval  = "regressor_dclaw"

    test = TestDClaw()
    test.load_model(group=group_model, model=model_model)
    test.load_evaluator(group=group_eval, model=model_eval)
    test.load_evaluation_dataset()
    (mean_yx, var_yx) = test.get_evaluator_predictive_distribution()
    (mean_y, var_y)   = test.get_evaluator_marginal_predictive_distribution(mean_yx, var_yx)

    inception_score   = metric.inception_score(var_yx=var_yx, var_y=var_y)
    H_yx              = metric.entropy_Hyx(var_yx)
    H_y               = metric.entropy_Hy(var_y)
