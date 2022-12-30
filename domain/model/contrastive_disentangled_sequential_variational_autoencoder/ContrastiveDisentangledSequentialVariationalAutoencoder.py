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
# from .loss.ContrastiveLoss import ContrastiveLoss
from .loss.loss import contrastive_loss, compute_mi
from custom import reparameterize
from .loss.MutualInformationFactory import MutualInformationFactory



class ContrastiveDisentangledSequentialVariationalAutoencoder(nn.Module):
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
        num_batch, step, channle, width, height = img.shape
        (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample) = self.encode(img)
        # f_mean_prior, f_logvar_prior                             = self.context_prior.dist(f_mean)
        z_mean_prior, z_logvar_prior, z_sample_prior               = self.motion_prior(z_sample)
        x_recon                                                    = self.decode(z_sample, f_sample)

        return  {
            "f_mean"         : f_mean,
            "f_logvar"       : f_logvar,
            "f_sample"       : f_sample,
            # "f_mean_prior"   : f_mean_prior,
            # "f_logvar_prior" : f_logvar_prior,
            "z_mean"         : z_mean,
            "z_logvar"       : z_logvar,
            "z_sample"       : z_sample,
            "z_mean_prior"   : z_mean_prior,
            "z_logvar_prior" : z_logvar_prior,
            "x_recon"        : x_recon
        }


    def kl_reverse(self, q, p, q_sample):
        kl = (q.log_prob(q_sample) - p.log_prob(q_sample))
        kl = kl.sum()
        return kl


    def define_normal_distribution(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mean, std)


    def loss_function(self,
                        x                        ,
                        batch_idx                ,
                        results_dict             ,
                        results_dict_aug_context ,
                        results_dict_aug_dynamics,
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

        f_mean_c      = results_dict_aug_context["f_mean"]
        z_post_mean_m = results_dict_aug_dynamics["z_mean"]

        batch_size, n_frame, z_dim = z_post_mean.size()

        mi_xs  = compute_mi(f, (f_mean, f_logvar))
        n_bs   = z_post.shape[0]

        mi_xzs = [compute_mi(z_post_t, (z_post_mean_t, z_post_logvar_t)) \
                    for z_post_t, z_post_mean_t, z_post_logvar_t in \
                    zip(z_post.permute(1,0,2), z_post_mean.permute(1,0,2), z_post_logvar.permute(1,0,2))]
        mi_xz  = torch.stack(mi_xzs).sum()

        # if opt.loss_recon == 'L2': # True branch
        l_recon = F.mse_loss(recon_x, x, reduction='sum')
        # else:
            # l_recon = torch.abs(recon_x - x).sum()

        f_mean   = f_mean.view((-1, f_mean.shape[-1])) # [128, 256]
        f_logvar = f_logvar.view((-1, f_logvar.shape[-1])) # [128, 256]
        kld_f    = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))

        z_post_var  = torch.exp(z_post_logvar) # [128, 8, 32]
        z_prior_var = torch.exp(z_prior_logvar) # [128, 8, 32]
        kld_z       = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                                ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

        l_recon, kld_f, kld_z = l_recon / batch_size, kld_f / batch_size, kld_z / batch_size

        con_loss_c = self.contrastive_loss(f_mean, f_mean_c)
        con_loss_m = self.contrastive_loss(z_post_mean.view(batch_size, -1), z_post_mean_m.view(batch_size, -1))

        # calculate the mutual infomation of f and z
        mi_fz = torch.zeros((1)).cuda()
        if True: # 0.1
            f_dist = (f_mean, f_logvar, f)
            z_dist = (z_post_mean, z_post_logvar, z_post)
            mi_fz  = self.mutual_information(f_dist=f_dist, z_dist=z_dist)
            # mi_fz, (Hf, Hz, Hfz), (value0_min_f, value0_max_f, value0_min_z, value0_max_z, value0_min_fz, value0_max_fz), (m_max_f, m_min_f, m_max_z, m_min_z, m_max_fz, m_min_fz) = self.mutual_information(f_dist=f_dist, z_dist=z_dist)

        # print("mi_fz = ", mi_fz.detach().cpu().numpy())
        # import ipdb; ipdb.set_trace()

        loss =  l_recon \
                + kld_f * self.weight.kld_context \
                + kld_z * self.weight.kld_dynamics \
                + mi_fz * self.weight.mutual_information_fz \
                + con_loss_c * self.weight.contrastive_loss_fx \
                + con_loss_m * self.weight.contrastive_loss_zx

        return {
            "loss"            : loss,
            "Train/mse"       : l_recon,
            "Train/kld_f"     : kld_f,
            "Train/kld_z"     : kld_z,
            "Train/con_loss_c": con_loss_c,
            "Train/con_loss_m": con_loss_m,
            "Train/mi_fz"     : mi_fz,

            # "Entropy/Hf"      : Hf,
            # "Entropy/Hz"      : Hz,
            # "Entropy/Hfz"     : Hfz,

            # "value0_minmax/f_min"  :   value0_min_f,
            # "value0_minmax/f_max"  :   value0_max_f,
            # "value0_minmax/z_min"  :   value0_min_z,
            # "value0_minmax/z_max"  :   value0_max_z,
            # "value0_minmax/fz_min" :   value0_min_fz,
            # "value0_minmax/fz_max" :   value0_max_fz,

            # "m_minmax/f_max" : m_max_f,
            # "m_minmax/f_min" : m_min_f,
            # "m_minmax/z_max" : m_max_z,
            # "m_minmax/z_min" : m_min_z,
            # "m_minmax/fz_max": m_max_fz,
            # "m_minmax/fz_min": m_min_fz,

            # "dist_minmax/f_mean_min"       : results_dict["f_mean"].min(),
            # "dist_minmax/f_mean_max"       : results_dict["f_mean"].max(),
            # "dist_minmax/f_logvar_min"     : results_dict["f_logvar"].min(),
            # "dist_minmax/f_logvar_max"     : results_dict["f_logvar"].max(),
            # "dist_minmax/z_post_mean_min"  : results_dict["z_mean"].min(),
            # "dist_minmax/z_post_mean_max"  : results_dict["z_mean"].max(),
            # "dist_minmax/z_post_logvar_min": results_dict["z_logvar"].min(),
            # "dist_minmax/z_post_logvar_max": results_dict["z_logvar"].max(),
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