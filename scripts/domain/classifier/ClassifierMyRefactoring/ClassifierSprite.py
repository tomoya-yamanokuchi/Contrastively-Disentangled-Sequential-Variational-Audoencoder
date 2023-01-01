import torch
from torch import nn
from typing import List
from omegaconf.omegaconf import OmegaConf
from .ClassifierUnit import ClassifierUnit
from .FullConvFrameEncoder import FullConvFrameEncoder as FrameEncoder


class ClassifierSprite(nn.Module):
    def __init__(self, config):
        super(ClassifierSprite, self).__init__()
        self.g_dim      = config.frame_encoder.dim_frame_feature
        self.channels   = config.frame_encoder.in_channels
        self.hidden_dim = config.bi_lstm_encoder.hidden_dim
        # self.frames     = config.frames

        self.frame_encoder = FrameEncoder(self.channels, self.g_dim)
        self.bilstm        = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_skin      = ClassifierUnit(self.hidden_dim, 6)
        self.cls_top       = ClassifierUnit(self.hidden_dim, 6)
        self.cls_pant      = ClassifierUnit(self.hidden_dim, 6)
        self.cls_hair      = ClassifierUnit(self.hidden_dim, 6)
        self.cls_action    = ClassifierUnit(self.hidden_dim, 9)


    def encoder_frame(self, x):
        '''
            input x is list of length Frames [batchsize, channels, size, size]
            convert it to [batchsize, frames, channels, size, size]
            x = torch.stack(x, dim=1)
            [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        '''
        x_shape = x.shape
        x       = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)


    def forward(self, x):
        import ipdb; ipdb.set_trace()
        conv_x      = self.encoder_frame(x)
        lstm_out, _ = self.bilstm(conv_x)
        backward    = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal     = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f  = torch.cat((frontal, backward), dim=1)
        return {
            'skin  ' : self.cls_skin(lstm_out_f),
            'top   ' : self.cls_top(lstm_out_f),
            'pant  ' : self.cls_pant(lstm_out_f),
            'hair  ' : self.cls_hair(lstm_out_f),
            'action' : self.cls_action(lstm_out_f),
        }

