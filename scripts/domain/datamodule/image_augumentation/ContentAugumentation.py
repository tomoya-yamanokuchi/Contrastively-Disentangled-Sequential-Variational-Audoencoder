import torch
from torch import Tensor


class ContentAugumentation:
    def augment(self, img: Tensor):
        '''
        - keep  : context
        - change: dynamics
        '''
        step         = img.shape[0]
        random_index = torch.randperm(step)
        return torch.index_select(img, dim=0, index=random_index)