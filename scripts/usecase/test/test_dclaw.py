
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
from domain.visualize.DClawDataPlot import DClawDataPlot
from custom import to_numpy
from custom import reparameterize
from custom import logsumexp
from custom import log_density_z


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
        # datamodule = DataModuleFactory().create(**self.config_eval.datamodule)
        # datamodule.setup(stage="fit")
        # self.train_dataloader = datamodule.train_for_test_dataloader()

        # << evaluation dataset >>
        datamodule = DataModuleFactory().create(**self.config_eval.datamodule)
        datamodule.setup(stage="test")
        self.test_dataloader = datamodule.test_dataloader()


    # def get_evaluator_predictive_distribution(self):
    #     mean   = []
    #     var    = []
    #     y_true = []
    #     for index, data in self.train_dataloader:
    #         x = data['input']  # image
    #         y = data['output'] # state

    #         recon_x_sample, recon_x = self.model.forward_fixed_motion_for_classification(x)
    #         results_dict = self.evaluator.forward(x)
    #         # results_dict = self.evaluator.forward(recon_x_sample)

    #         # convert from tensor to numpy
    #         y             = to_numpy(y)
    #         mean_ensemble = to_numpy(results_dict["mean"])
    #         var_ensemble  = to_numpy(results_dict["var"])

    #         # calculate predictive distribution
    #         mean_pred = mean_ensemble.mean(axis=0)
    #         var_pred  = (var_ensemble + mean_ensemble**2).mean(axis=0) - mean_pred**2

    #         mean.append(mean_pred)
    #         var.append(var_pred)
    #         y_true.append(y)

    #     mean        = np.concatenate(mean, axis=0)
    #     var         = np.concatenate(var, axis=0)
    #     self.y_true = np.concatenate(y_true, axis=0)

    #     # ----------------------
    #     # import ipdb; ipdb.set_trace()
    #     # num_batch, step, dim_y = mean.shape
    #     # x = np.linspace(1, step, step)

    #     # # << plot data: indivisdual mean and variance >>
    #     # sin_plot = DClawDataPlot(xlabel="x", ylabel="y", title="y=sin(x)")
    #     # # sin_plot.plot_true_function(x, y[index])
    #     # for n in range(mean.shape[0]):
    #     #     # import ipdb; ipdb.set_trace()
    #     #     sin_plot.plot_prediction(x, mean[n], np.sqrt(var[n]))
    #     # sin_plot.save_fig('./all_train_mean_var.png')
    #     # import ipdb; ipdb.set_trace()
    #     # ----------------------
    #     return (mean, var)


    def log_marginal_predictive_density(self, log_density_yx):
        # import ipdb; ipdb.set_trace()
        logsumexp_p_yx = logsumexp(torch.Tensor(log_density_yx), dim=0, keepdim=False)
        log_py         = to_numpy(logsumexp_p_yx) - np.log(mean_yx.shape[0])
        return log_py


    # def log_marginal_predictive_density(self, mean_yx, var_yx, num_sampling=100):
    #     log_py_samples = [self._log_marginal_predictive_density(mean_yx, var_yx) for i in range(num_sampling)]
    #     return np.array(log_py_samples)


    def log_gaussian_density(self, mean, var, sample):
        dim         = mean.shape[-1]
        c           = dim * np.log(2*np.pi)
        y           = sample - mean
        y_sigma     = y / np.sqrt(var)
        log_density = -0.5 * (c + np.log(var).sum(axis=-1) + (y_sigma**2).sum(axis=-1))
        return log_density.sum(-1) # sum over timestep


    def get_log_gaussian_density(self, mean_yx, var_yx):
        sample_yx      = self.reparameterize(mean_yx, np.log(var_yx))
        log_density_yx = self.log_gaussian_density(mean_yx, var_yx, sample_yx)
        return log_density_yx


    def evaluate(self):
        z_mean_x     = []
        z_logvar_x   = []

        z_mean_gen   = []
        z_logvar_gen = []

        ensemble_mean = []
        ensemble_var  = []
        y_true        = []
        for index, data in self.test_dataloader:
            x = data['input']  # image
            y = data['output'] # state
            # -------------------------------------------------
            #             get motion dist q(z|x)
            # -------------------------------------------------
            # << encode original data >>
            _, (z_mean, z_logvar, z_sample) = self.model.encode(x)
            z_mean_x.append(to_numpy(z_mean))
            z_logvar_x.append(to_numpy(z_logvar))

            # << generate new data and encode again>>
            x_gen = self.model.forward_fixed_motion(z_sample)
            _, (z_mean, z_logvar, z_sample) = self.model.encode(x_gen)
            z_mean_gen.append(to_numpy(z_mean))
            z_logvar_gen.append(to_numpy(z_logvar))

            # -------------------------------------------------
            #              ensemble prediction
            # -------------------------------------------------
            result_ensemble = self.evaluator.forward(x_gen)
            y               = to_numpy(y)
            mean_ensemble   = to_numpy(result_ensemble["mean"])
            var_ensemble    = to_numpy(result_ensemble["var"])
            # << calculate predictive distribution >>
            mean_pred       = mean_ensemble.mean(axis=0)
            var_pred        = (var_ensemble + mean_ensemble**2).mean(axis=0) - mean_pred**2
            # << append data >>
            ensemble_mean.append(mean_pred)
            ensemble_var.append(var_pred)
            y_true.append(y)
        # -------------------------------------------------
        #              ensemble prediction
        # -------------------------------------------------
        z_mean_x      = np.concatenate(z_mean_x  , axis=0)
        z_logvar_x    = np.concatenate(z_logvar_x, axis=0)

        z_mean_gen    = np.concatenate(z_mean_gen  , axis=0)
        z_logvar_gen  = np.concatenate(z_logvar_gen, axis=0)

        ensemble_mean = np.concatenate(ensemble_mean, axis=0)
        ensemble_var  = np.concatenate(ensemble_var, axis=0)
        y_true        = np.concatenate(y_true, axis=0)
        return {
            "z_mean_x"     : z_mean_x,
            "z_logvar_x"   : z_logvar_x,
            "z_mean_gen"   : z_mean_gen,
            "z_logvar_gen" : z_logvar_gen,
            "ensemble_mean": ensemble_mean,
            "ensemble_var" : ensemble_var,
            "y_true"       : y_true,
        }


if __name__ == '__main__':

    model_model = "[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[300epoch]-[20230105142322]-melco_dclaw_config_cdsvae_dclaw"
    group_model = "cdsvae_dclaw"

    model_eval  = "[regressor_dclaw]-[action_norm_valve]-[dim_out=8]-[dim_fc_hidden=256]-[100epoch]-[num_batch=128]-[beta=0.5]-[20230109222005]-[melco]-"
    group_eval  = "regressor_dclaw"

    test        = TestDClaw()
    test.load_model(group=group_model, model=model_model)
    test.load_evaluator(group=group_eval, model=model_eval)
    test.load_evaluation_dataset()


    loss_list = []
    kl_list   = []
    H_y_list  = []
    for i in range(5):
        result_dict = test.evaluate()

        loss = (result_dict["y_true"] - result_dict["ensemble_mean"]).sum(-1).mean(axis=-1).mean(axis=-1)

        kl = metric.kl_divergence(
            q = (result_dict["z_mean_gen"], result_dict["z_logvar_gen"]),
            p = (result_dict["z_mean_x"], result_dict["z_logvar_x"]),
        )

        # << ensemble entropy>>
        ensemble_mean   = result_dict["ensemble_mean"]
        ensemble_var    = result_dict["ensemble_var"]
        sample_p_yx     = reparameterize(mean=torch.Tensor(ensemble_mean), logvar=torch.Tensor(np.log(ensemble_var)))
        sample_p_yx     = to_numpy(sample_p_yx)
        N               = ensemble_mean.shape[0]
        log_p_yx_matrix = log_density_z(mean=torch.Tensor(ensemble_mean), logvar=torch.Tensor(np.log(ensemble_var)), sample=torch.Tensor(sample_p_yx))
        logsumexp_p_y   = logsumexp(log_p_yx_matrix, dim=-1, keepdim=True) # sum over inner minibach (index j)
        H_y             = - torch.mean(logsumexp_p_y.squeeze() - torch.Tensor([np.log(N*N)]))

        print("[loss, kl, Hy] = [{:.3f}, {:.2f}, {:.2f}]".format(loss, kl, H_y))

        loss_list.append(loss)
        kl_list.append(kl)
        H_y_list.append(H_y)

    print("-------------------    mean   ------------------------")
    print("[loss, kl, Hy] = [{:.3f}, {:.2f}, {:.2f}]".format(np.mean(loss_list), np.mean(kl_list), np.mean(H_y_list)))
    print("------------------------------------------------------")