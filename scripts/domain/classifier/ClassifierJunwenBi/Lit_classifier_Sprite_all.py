import os
# import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import numpy as np
import torch
import pytorch_lightning as pl
from .classifier_Sprite_all import classifier_Sprite_all


class Lit_classifier_Sprite_all(pl.LightningModule):
    def __init__(self,
                 config,
                 num_train) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.config       = config
        self.num_train    = num_train
        self.model        = classifier_Sprite_all(config.model)
        self.summary_dict = None
        # self.summary = torchinfo.summary(self.model, input_size=(131, 8, 3, 64, 64))


    def forward(self, input, **kwargs):
        return self.model.forward(input)
