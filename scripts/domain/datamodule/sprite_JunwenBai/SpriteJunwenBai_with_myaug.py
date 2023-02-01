import pickle
import numpy as np
import torch
from torchvision.datasets import VisionDataset
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from ..image_augumentation.ContentAugumentation import ContentAugumentation
from ..image_augumentation.MotionAugumentation import MotionAugumentation



class SpriteJunwenBai_with_myaug(VisionDataset):
    ''' Sprite Dataset (shared by Junwen Bai)
        - sequence
            - train: 9000
            - test : 2664
        - step              : 8
        - image size        : (3, 64, 64)
        - motion variation  : 9
            - actions (walking, spellcasting, slashing) x directions (left, front, right)
        - minmax value:
            - min:  0.0
            - max:  1.0
    '''

    def __init__(self, data_dir: str, train: bool):
        self.data_dir              = data_dir
        self.train                 = train
        self.min                   = 0.0
        self.max                   = 1.0
        self.content_augumentation = ContentAugumentation()
        self.motion_augumentation  = MotionAugumentation()
        self.__all_preload_and_preprocessing()


    def __len__(self):
        return self.N


    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)


    def _image_range_assetion(self, img):
        assert img.min() >= self.min, print("[img.min(), self.min] = [{}, {}]".format(img.min(), self.min))
        assert img.max() <= self.max, print("[img.max(), self.max] = [{}, {}]".format(img.max(), self.max))
        return img


    def __all_preload_and_preprocessing(self):
        if   self.train: name = "train"
        else           : name = "test"
        # --------- load dataset ---------
        data          = pickle.load(open(self.data_dir + '/{}.pkl'.format(name), 'rb'))
        A_label_anker = data['A_{}'.format(name)]
        D_label_anker = data['D_{}'.format(name)]
        data          = data['X_{}'.format(name)] # destructive assgin
        self.N        = data.shape[0]
        # ------ apply augumentation ------
        self.images  = []
        self.c_aug   = []
        self.m_aug   = []
        self.A_label = []
        self.D_label = []
        for n in range(self.N):
            img_torch = self.to_tensor_image(data[n]) # (8, 3, 64, 64)
            self.images.append(self._image_range_assetion(img_torch))
            self.c_aug.append( self._image_range_assetion(self.content_augumentation.augment(img_torch)))
            self.m_aug.append( self._image_range_assetion(self.motion_augumentation.augment(img_torch)))
            self.A_label.append(self.to_tensor_label(A_label_anker[n]))
            self.D_label.append(self.to_tensor_label(D_label_anker[n]))
        self.images  = torch.stack(self.images,  axis=0)
        self.c_aug   = torch.stack(self.c_aug,   axis=0)
        self.m_aug   = torch.stack(self.m_aug,   axis=0)
        self.A_label = torch.stack(self.A_label, axis=0)
        self.D_label = torch.stack(self.D_label, axis=0)


    def __getitem__(self, index):
        return index, {
            "images" : self.images[index].cuda(),
            "c_aug"  : self.c_aug[index].cuda(),
            "m_aug"  : self.m_aug[index].cuda(),
            "A_label": self.A_label[index].cuda(),
            "D_label": self.D_label[index].cuda(),
            "index"  : index,
        }


    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()


    def to_tensor_label(self, label):
        # import ipdb; ipdb.set_trace()
        # assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.as_tensor(label.astype('float32'))
