import numpy as np
from custom import assert_range, normalize, to_tensor



def get_state_minmax(mode):
    '''
    データセット全体から予め計算しておく必要あり
    '''
    state_minmax = {
        'joint_space_position'    : (-0.43635426000464594, 0.3996889727516365),
        'task_space_positioin'    : (0.0, 1.0),
    }
    return state_minmax[mode]



class State2Tensor:
    def __init__(self, robot_state):
        self.robot_state   = robot_state
        self.min, self.max = get_state_minmax(robot_state)

    def get(self, db):
        robot_state  = self._get_robot_state(db["state"][self.robot_state])
        object_state = self._get_object_state(db["state"]["object_position"])
        state        = np.concatenate((robot_state, object_state), axis=-1)
        assert_range(state, 0, 1)
        return to_tensor(state)


    def _get_robot_state(self, robot_state):
        assert self.robot_state == "task_space_positioin", print("task_space_positioin じゃないとsin, cosの変換がおかしくなる")
        assert_range(robot_state, self.min, self.max)
        robot_state = self.convert_task_space_position(robot_state)
        return robot_state


    def _get_object_state(self, object_state):
        object_state = self.convert_valve_angle(object_state)
        assert_range(x=object_state, min=0, max=1)
        return object_state


    def convert_task_space_position(self, task_space_position):
        assert_range(task_space_position, 0, 1)
        theta = normalize(task_space_position, x_min=0, x_max=1, m=0, M=2*np.pi)
        x     = np.cos(theta)
        y     = np.sin(theta)

        x_reshaped = np.transpose(x)[np.newaxis, :, :]
        y_reshaped = np.transpose(y)[np.newaxis, :, :]

        xy_reshaped = np.concatenate((x_reshaped, y_reshaped), axis=0)
        xy_reshaped = xy_reshaped.reshape(-1, xy_reshaped.shape[-1], order="F") # xとyの次元が交互にならぶようにreshapeする
        xy_reshaped = normalize(xy_reshaped, x_min=-1, x_max=1, m=0, M=1)
        return xy_reshaped.transpose()


    def convert_valve_angle(self, valve_radian):
        theta = valve_radian
        x     = np.cos(3*theta)
        y     = np.sin(3*theta)
        # normaize
        xy = np.concatenate((x, y), axis=-1)
        xy = normalize(xy, x_min=-1, x_max=1, m=0, M=1)
        return xy
