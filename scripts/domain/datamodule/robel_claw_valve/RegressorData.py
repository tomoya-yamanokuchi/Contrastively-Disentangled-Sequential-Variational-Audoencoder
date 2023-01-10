import torch
import numpy as np
from pprint import pprint
from torchvision.datasets import VisionDataset
import shelve
from natsort import natsorted
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from custom.utility.normalize import normalize


def get_state_minmax(mode):
    '''
    データセット全体から予め計算しておく必要あり
    '''
    state_minmax = {
        'joint_space_position'    : (-0.43635426000464594, 0.3996889727516365),
        'task_space_positioin'    : (0.0, 1.0),
    }
    return state_minmax[mode]


class RegressorData(VisionDataset):
    ''' Sprite Dataset
        - sequence
            - train: 2000
            - test : 200
        - step              : 25
        - image size        : (3, 64, 64)
        - action variation  : 8
            - claw1だけ動く
            - claw2だけ動く
            - claw3だけ動く
            - ３本同時に動く（左右）など
        - minmax value:
            - min: -1.0
            - max:  1.0
    '''

    def __init__(self, data_dir: str, train: bool, data_type: dict):
        self.data_dir              = data_dir
        self.train                 = train
        self.data_type             = data_type
        self.img_paths             = self._get_img_paths()
        self.num_data              = len(self.img_paths)
        self.min                   = 0
        self.max                   = 255
        self.state_min, self.state_max = get_state_minmax(data_type["robot_state"])
        self.__all_preload()


    def _get_img_paths(self):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = self.data_dir + "/dataset_202210221514_valve2000_train/"
        else         : img_dir = self.data_dir + "/dataset_20221022153117_valve200_test/"
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix == ".db"]
        img_paths = natsorted(img_paths)
        # import ipdb; ipdb.set_trace()
        return img_paths


    def assert_range(self, x, min, max):
        assert x.min() >= min, print("[x.min(), min] = [{}, {}]".format(x.min(), min))
        assert x.max() <= max, print("[x.max(), max] = [{}, {}]".format(x.max(), max))
        return x



    def __all_preload(self):
        self.images = []
        self.state  = []
        for path in self.img_paths:
            path_without_suffix = str(path.resolve()).split(".")[0]          #; print(path_without_suffix)
            db                  = shelve.open(path_without_suffix, flag='r') # read only

            # ------ image ------
            img_numpy           = db["image"]["canonical"]                   # 複数ステップ分が含まれている(1系列分)
            self.assert_range(img_numpy, self.min, self.max)
            step, width, height, channel = img_numpy.shape                   # channlの順番に注意（保存形式に依存する）
            assert channel == 3
            img_torch = self.to_tensor_image(img_numpy)
            img_torch = normalize(x=img_torch, x_min=self.min, x_max=self.max, m=0, M=1)
            self.images.append(img_torch)

            # ------ robot state ------
            state        = db["state"]
            assert self.data_type["robot_state"] == "task_space_positioin", print("task_space_positioin じゃないとsin, cosの変換がおかしくなる")
            robot_state  = state[self.data_type["robot_state"]]
            robot_state  = self.assert_range(robot_state, self.state_min, self.state_max)
            robot_state = self.convert_task_space_position(robot_state)

            # ------ object state ------
            object_state = self.convert_valve_angle(state["object_position"])
            self.assert_range(object_state, 0, 1)

            # ------ all state ------
            # import ipdb; ipdb.set_trace()
            # all_state    = np.concatenate(
            #     (robot_state_x, object_state_x,
            #      robot_state_y, object_state_y),
            #     axis=-1
            # )
            all_state    = np.concatenate((robot_state, object_state), axis=-1)
            all_state    = self.assert_range(all_state, 0, 1)
            state_tensor = self.to_tensor_state(all_state)
            self.state.append(state_tensor)
            # import ipdb; ipdb.set_trace()

        self.images = torch.stack(self.images, axis=0)
        self.state  = torch.stack(self.state, axis=0)
        # import ipdb; ipdb.set_trace()


    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)


    def __getitem__(self, index: int):
        return index, {
            "input"      : self.images[index].cuda(),
            "output"     : self.state[index].cuda(),
            "index"      : index,
        }

    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()

    def to_tensor_state(self, state):
        assert len(state.shape) == 2 # (step, dim_u)
        return torch.from_numpy(state).contiguous().type(torch.FloatTensor)

    def to_tensor_label(self, label):
        # assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(label).contiguous()


    # def convert_valve_angle(self, valve_radian):
    #     theta = valve_radian
    #     x     = np.cos(3*theta)
    #     y     = np.sin(3*theta)
    #     # normaize
    #     x    = normalize(x, x_min=-1, x_max=1, m=0, M=1)
    #     y    = normalize(y, x_min=-1, x_max=1, m=0, M=1)
    #     # import ipdb; ipdb.set_trace()
    #     return (x, y)


    def convert_valve_angle(self, valve_radian):
        theta = valve_radian
        x     = np.cos(3*theta)
        y     = np.sin(3*theta)
        # normaize
        xy = np.concatenate((x, y), axis=-1)
        xy = normalize(xy, x_min=-1, x_max=1, m=0, M=1)
        return xy


    # def convert_task_space_position(self, task_space_position):
    #     self.assert_range(task_space_position, 0, 1)
    #     theta = normalize(task_space_position, x_min=0, x_max=1, m=0, M=2*np.pi)
    #     x     = np.cos(theta)
    #     y     = np.sin(theta)
    #     # xy    = np.concatenate((x, y), axis=-1)
    #     x     = normalize(x, x_min=-1, x_max=1, m=0, M=1)
    #     y     = normalize(y, x_min=-1, x_max=1, m=0, M=1)
    #     # import ipdb; ipdb.set_trace()
    #     return (x, y)


    def convert_task_space_position(self, task_space_position):
        self.assert_range(task_space_position, 0, 1)
        theta = normalize(task_space_position, x_min=0, x_max=1, m=0, M=2*np.pi)
        x     = np.cos(theta)
        y     = np.sin(theta)

        x_reshaped = np.transpose(x)[np.newaxis, :, :]
        y_reshaped = np.transpose(y)[np.newaxis, :, :]

        xy_reshaped = np.concatenate((x_reshaped, y_reshaped), axis=0)
        xy_reshaped = xy_reshaped.reshape(-1, xy_reshaped.shape[-1], order="F") # xとyの次元が交互にならぶようにreshapeする
        xy_reshaped = normalize(xy_reshaped, x_min=-1, x_max=1, m=0, M=1)
        return xy_reshaped.transpose()
