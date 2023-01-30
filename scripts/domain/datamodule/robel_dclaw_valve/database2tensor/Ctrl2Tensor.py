import numpy as np
from custom import assert_range, normalize, to_tensor



def get_uminmax(mode):
    '''
    データセット全体から予め計算しておく必要あり
    '''
    u_minmax = {
        'task_space_diff_position': (-0.050, 0.050),
        'joint_space_position'    : (-0.43635426000464594, 0.3996889727516365),
    }
    return u_minmax[mode]



class Ctrl2Tensor:
    def __init__(self, ctrl_type):
        self.ctrl_type     = ctrl_type
        self.min, self.max = get_uminmax(ctrl_type)


    def get(self, db):
        ctrl = db["ctrl"][self.ctrl_type]
        # ctrl = self.extract_active_joint(ctrl)
        # self.assert_range_2dim(ctrl, self.u_min.reshape(1, -1), self.u_max.reshape(1, -1))
        assert ctrl.min() >= self.min, print("ctrl.min() >= self.u_min = [{}, {}]".format(ctrl.min(), self.min))
        assert ctrl.max() <= self.max, print("ctrl.max() >= self.u_max = [{}, {}]".format(ctrl.max(), self.max))
        ctrl = normalize(x=ctrl, x_min=self.min, x_max=self.max, m=0, M=1)
        return to_tensor(ctrl)


    def extract_active_joint(self, ctrl):
        assert len(ctrl.shape) == 2
        # import ipdb; ipdb.set_trace()
        ctrl = np.take(ctrl, [0, 1, 3, 4, 6, 7], axis=-1)
        return ctrl


    def assert_range_2dim(self, x, min, max):
        # import ipdb; ipdb.set_trace()
        eps  = -1e-7
        diff = x - min; assert (diff < eps).sum() < 1
        diff = max - x; assert (diff < eps).sum() < 1