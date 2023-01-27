import torch
from torch import nn
from torch import Tensor
from typing import List
from torch.nn import functional as F
from omegaconf.omegaconf import OmegaConf
from .inference_model.frame_encoder.FrameEncoderFactory import FrameEncoderFactory
from .inference_model.ContextEncoder import ContextEncoder
from .inference_model.motion_encoder.MotionEncoderFactory import MotionEncoderFactory
from .inference_model.BiLSTMEncoder import BiLSTMEncoder
from .generative_model.frame_decoder.FrameDecoderFactory import FrameDecoderFactory
from .generative_model.MotionPrior import MotionPrior
from .generative_model.ContentPrior import ContentPrior
from .loss.loss import contrastive_loss, compute_mi
from .loss.MutualInformationFactory import MutualInformationFactory



class ContrastivelyDisentangledSequentialVariationalAutoencoder(nn.Module):
    def __init__(self,
                 network    : OmegaConf,
                 loss       : OmegaConf,
                 num_train  : int,
                 **kwargs) -> None:
        super().__init__()
        # inference
        self.frame_encoder      = FrameEncoderFactory().create(**network.frame_encoder)
        self.bi_lstm_encoder    = BiLSTMEncoder(in_dim=network.frame_encoder.dim_frame_feature, **network.bi_lstm_encoder)
        self.context_encoder    = ContextEncoder(lstm_hidden_dim=network.bi_lstm_encoder.hidden_dim, **network.context_encoder)
        self.motion_encoder     = MotionEncoderFactory().create(lstm_hidden_dim=network.bi_lstm_encoder.hidden_dim, **network.motion_encoder)
        # prior
        self.content_prior      = ContentPrior(network.context_encoder.context_dim)
        self.motion_prior       = MotionPrior(**network.motion_prior)
        # generate
        in_dim_decoder          = network.context_encoder.context_dim + network.motion_encoder.state_dim
        self.frame_decoder      = FrameDecoderFactory().create(**network.frame_decoder, in_dim=in_dim_decoder, out_channels=network.frame_encoder.in_channels)
        # loss
        self.weight             = loss.weight
        self.contrastive_loss   = contrastive_loss(**loss.contrastive_loss)
        self.mutual_information = MutualInformationFactory().create(**loss.mutual_information, num_train=num_train)


    def encode(self, img):
        encoded_frame              = self.frame_encoder(img)             # shape = [num_batch, step, conv_fc_out_dims[-1]]
        bi_lstm_out                = self.bi_lstm_encoder(encoded_frame)
        f_mean, f_logvar, f_sample = self.context_encoder(bi_lstm_out)   # both shape = [num_batch, context_dim]
        z_mean, z_logvar, z_sample = self.motion_encoder(bi_lstm_out)    # both shape = [num_batch, step, state_dim]
        return (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample)


    def decode(self, z, f):
        num_batch, step, _ = z.shape
        return self.frame_decoder(torch.cat((z, f.unsqueeze(1).expand(num_batch, step, -1)), dim=2))


    def forward(self, img: Tensor, **kwargs) -> List[Tensor]:
        # num_batch, step, channle, width, height = img.shape
        (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample) = self.encode(img)
        z_mean_prior, z_logvar_prior, z_sample_prior               = self.motion_prior(z_sample)
        x_recon                                                    = self.decode(z_sample, f_sample)
        return  {
            "f_mean"         : f_mean,
            "f_logvar"       : f_logvar,
            "f_sample"       : f_sample,
            "z_mean"         : z_mean,
            "z_logvar"       : z_logvar,
            "z_sample"       : z_sample,
            "z_mean_prior"   : z_mean_prior,
            "z_logvar_prior" : z_logvar_prior,
            "x_recon"        : x_recon
        }


    def loss_function(self,
                        x                        ,
                        batch_idx                ,
                        results_dict             ,
                        results_dict_aug_context ,
                        results_dict_aug_dynamics,
                        step_mode,
                        **kwargs) -> dict:

        f_mean         = results_dict["f_mean"]
        f_logvar       = results_dict["f_logvar"]
        f              = results_dict["f_sample"]
        z_post_mean    = results_dict["z_mean"]
        z_post_logvar  = results_dict["z_logvar"]
        z_post         = results_dict["z_sample"]
        z_prior_mean   = results_dict["z_mean_prior"]
        z_prior_logvar = results_dict["z_logvar_prior"]
        recon_x        = results_dict["x_recon"]
        f_mean_c       = results_dict_aug_context["f_mean"]
        z_post_mean_m  = results_dict_aug_dynamics["z_mean"]

        batch_size = z_post_mean.size(0)

        l_recon  = F.mse_loss(recon_x, x, reduction='sum')

        f_mean   = f_mean.view((-1, f_mean.shape[-1]))
        f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))
        kld_f    = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))

        z_post_var  = torch.exp(z_post_logvar)
        z_prior_var = torch.exp(z_prior_logvar)
        kld_z       = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                                ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

        l_recon = l_recon / batch_size
        kld_f   =   kld_f / batch_size
        kld_z   =   kld_z / batch_size

        con_loss_c = self.contrastive_loss(f_mean, f_mean_c)
        con_loss_m = self.contrastive_loss(z_post_mean.view(batch_size, -1), z_post_mean_m.view(batch_size, -1))

        f_dist               = (f_mean, f_logvar, f)
        z_dist               = (z_post_mean, z_post_logvar, z_post)
        (Hf, Hz, Hfz), mi_fz = self.mutual_information(f_dist=f_dist, z_dist=z_dist)

        loss =  l_recon \
                + kld_f * self.weight.kld_context \
                + kld_z * self.weight.kld_dynamics \
                + mi_fz * self.weight.mutual_information_fz \
                + con_loss_c * self.weight.contrastive_loss_fx \
                + con_loss_m * self.weight.contrastive_loss_zx

        return {
            "{}/loss/loss"        .format(step_mode) : loss,
            "{}/loss/l_recon"     .format(step_mode) : l_recon,
            "{}/loss/kld_f"       .format(step_mode) : kld_f,
            "{}/loss/kld_z"       .format(step_mode) : kld_z,
            "{}/loss/con_loss_c"  .format(step_mode) : con_loss_c,
            "{}/loss/con_loss_m"  .format(step_mode) : con_loss_m,
            "{}/loss/mi_fz"       .format(step_mode) : mi_fz,
            # --- add ---
            "{}/Entropy/Hf"       .format(step_mode) : Hf,
            "{}/Entropy/Hz"       .format(step_mode) : Hz,
            "{}/Entropy/Hfz"      .format(step_mode) : Hfz,
        }


    def forward_fixed_motion_for_classification(self, img):
        num_batch            = img.shape[0]
        (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample) = self.encode(img)
        _, _, f_prior_sample = self.content_prior.sample(num_batch)
        recon_x_sample       = self.decode(z_sample, f_prior_sample)
        recon_x              = self.decode(z_sample, f_mean)
        return recon_x_sample, recon_x


    def forward_fixed_motion(self, z):
        num_batch      = z.shape[0]
        _, _, f_sample = self.content_prior.sample(num_batch)
        return self.decode(z, f_sample)


    def forward_fixed_content(self, f, step):
        num_batch    = f.shape[0]
        z_mean, _, z_sample = self.motion_prior.sample(num_batch, step)
        return self.decode(z_sample, f)