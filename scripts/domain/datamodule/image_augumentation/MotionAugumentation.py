import torch
from torch import Tensor
from torchvision import transforms
from custom import normalize


class MotionAugumentation:
    '''
    - keep  : dynamics
    - change: context

    combination of
        - (cropping)
        - color distortion
        - Gaussian blur
        - reshaping
    '''
    def __init__(self):
        self.transform_color_dist   = self.get_transform_color_distortion()
        self.transform_GaussianBlur = self.get_transform_GaussianBlur()


    def augment(self, img: Tensor):
        '''
        image is a tensor with value range in [0, 1].
        '''
        assert (img.min() >= 0.0) and (img.max() <= 1.0)
        img = self.transform_color_dist(img)
        img = torch.cat([self.transform_GaussianBlur(_img) for _img in torch.split(img, 1, 0)], dim=0)
        return img


    def get_transform_color_distortion(self, s=1.0):
        '''
        - s is the strength of color distortion.
        '''
        color_jitter = transforms.ColorJitter(
            brightness = 0.8*s,
            contrast   = 0.8*s,
            saturation = 0.8*s,
            hue        = 0.2*s,
        )
        rnd_color_jitter = transforms.RandomApply(transforms=[color_jitter], p=0.8)
        rnd_gray         = transforms.RandomGrayscale(p=0.2)
        return transforms.Compose([rnd_color_jitter, rnd_gray])


    def get_transform_GaussianBlur(self, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        '''
        - kernel_size: set to be 10% of the image height/width
            (if img_size = (64, 64) -> kernel_size=(7, 7) with odd restriction).
        - sigma      : randomly sampled from [0.1, 2.0]
          (Please refere [Ting Chen, et al., ICML2020])
        '''
        assert type(kernel_size) == tuple
        assert type(sigma) == tuple
        blurrer     = transforms.GaussianBlur(kernel_size, sigma)
        rnd_blurrer = transforms.RandomApply(transforms=[blurrer], p=0.5)
        return transforms.Compose([rnd_blurrer])
