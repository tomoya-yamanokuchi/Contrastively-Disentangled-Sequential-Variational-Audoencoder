
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
from custom import save_image


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




    def evaluate(self):
        z_mean_x     = []
        z_logvar_x   = []

        z_mean_gen   = []
        z_logvar_gen = []

        ensemble_mean = []
        ensemble_var  = []
        y_true        = []
        for index, data in self.test_dataloader:
            x = data['images']
            y = data['state']

            y = to_numpy(y)

            # -------------------------------------------------
            #             get motion dist q(z|x)
            # -------------------------------------------------
            num_f_sampling = 25
            for i in range(num_f_sampling):
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

        # ----------------------
        #     plot ensemble
        # ----------------------
        _, step, dim_y = ensemble_mean.shape
        xx             = np.linspace(1, step, step)
        sin_plot       = DClawDataPlot(xlabel="x", ylabel="y", title="y=sin(x)")
        # import ipdb; ipdb.set_trace()
        for n in range(self.test_dataloader.dataset.num_data):
            sin_plot.plot_prediction(xx, ensemble_mean[n], np.sqrt(ensemble_var[n]))
        sin_plot.save_fig('./all_train_mean_var.png')
        # import ipdb; ipdb.set_trace()
        # ----------------------
        num_save_image = 8*4
        save_image(save_path="./x.png", image=x[:num_save_image])
        save_image(save_path="./x_gen.png", image=x_gen[:num_save_image])

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
    # model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[300epoch]-[20230105142322]-melco_dclaw_config_cdsvae_dclaw"
    # model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[300epoch]-[20230105144151]-remote3090_dclaw_config_cdsvae_dclaw"

    # search_key = "melco"
    # search_key = "remote3090"
    # search_key = "remote_tsukumo3090ti"
    # search_key = "remote_3090"

    # search_key = "[remote_tsukumo3090ti]-unique_content"
    # search_key = "[remote_3090]-unique_content"

    # search_key   = "[remote_3090]-mmm"
    # search_key   = "[remote_tsukumo3090ti]-mmm"

    # search_key = "[remote_3090]-unique_content_s"
    # search_key = "[remote_tsukumo3090ti]-unique_content_s"

    # search_key = "[remote_3090]-www"
    search_key = "[remote_tsukumo3090ti]-www"

    group_model       = "cdsvae_dclaw_deterministic"
    pathlib_obj       = pathlib.Path("/hdd_mount/logs_cdsvae/{}".format(group_model))
    model_cdsvae_list = [str(model).split("/")[-1] for model in list(pathlib_obj.glob("*")) if search_key in str(model)]

    num_eval_per_model = 1

    loss_total = []
    kl_total   = []
    Hyx_total  = []
    H_y_total  = []

    for m, model_cdsvae in enumerate(model_cdsvae_list):

        model_eval  = "[regressor_dclaw]-[robel_dclaw_deterministic]-[dim_out=8]-[dim_fc_hidden=256]-[30epoch]-[num_batch=128]-[beta=0.5]-[20230128120724]-[dl-box]-"
        group_eval  = "regressor_dclaw_deterministic"

        test        = TestDClaw()
        test.load_model(group=group_model, model=model_cdsvae)
        test.load_evaluator(group=group_eval, model=model_eval)
        test.load_evaluation_dataset()

        loss_list = []
        Hyx_list  = []
        kl_list   = []
        H_y_list  = []
        for i in range(num_eval_per_model):
            result_dict = test.evaluate()

            loss = ((result_dict["y_true"] - result_dict["ensemble_mean"])**2).sum(-1).mean(axis=-1).mean(axis=-1) # L2-norm

            Hyx = metric.entropy_Hyx(var_yx=result_dict["ensemble_var"])

            kl = metric.kl_divergence(
                q = (result_dict["z_mean_gen"], result_dict["z_logvar_gen"]),
                p = (result_dict["z_mean_x"], result_dict["z_logvar_x"]),
            )

            # << ensemble entropy>>
            ensemble_mean   = torch.Tensor(result_dict["ensemble_mean"])
            ensemble_var    = torch.Tensor(result_dict["ensemble_var"])
            sample_p_yx     = reparameterize(mean=ensemble_mean, logvar=ensemble_var.log())
            N               = ensemble_mean.shape[0]
            log_p_yx_matrix = log_density_z(mean=ensemble_mean, logvar=ensemble_var.log(), sample=sample_p_yx)
            logsumexp_p_y   = logsumexp(log_p_yx_matrix, dim=-1, keepdim=True) # sum over inner minibach (index j)
            H_y             = - torch.mean(logsumexp_p_y.squeeze() - torch.Tensor([np.log(N*N)]))

            # print("[Error↓ , KL↓ , H(y)↑ ] = [{:.3f}, {:.2f}, {:.2f}]".format(loss, kl, H_y))

            loss_list.append(loss)
            Hyx_list.append(Hyx)
            kl_list.append(kl)
            H_y_list.append(H_y)
        # ------------------------------------------
        # import ipdb; ipdb.set_trace()
        loss_mean        = np.mean(loss_list)
        Hyx_mean         = np.mean(Hyx_list)
        kl_mean          = np.mean(kl_list)
        H_y_mean         = np.mean(H_y_list)
        print("     (model {}/{}) [Error↓ , KL↓ , H(y|x)↓ , H(y)↑ ] = [{:.3f}, {:.3f}, {:.2f}, {:.2f}] : {}".format(
            m+1, len(model_cdsvae_list), loss_mean, kl_mean, Hyx_mean, H_y_mean, model_cdsvae))

        loss_total.append(loss_mean)
        Hyx_total.append(Hyx_mean)
        kl_total.append(kl_mean)
        H_y_total.append(H_y_mean)

    print("-----------------------------------------------------------------------")
    print(" total mean (M={}) [Error↓ , KL↓ , H(y|x)↓ , H(y)↑ ] = [{:.3f} & {:.3f} & {:.2f} & {:.2f}]".format(
        len(model_cdsvae_list), np.mean(loss_total), np.mean(kl_total), np.mean(Hyx_total), np.mean(H_y_total)))
    print("-----------------------------------------------------------------------")


