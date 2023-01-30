import numpy as np
from custom import assert_range, normalize, to_tensor



def get_state_minmax(mode):
    '''
    データセット全体から予め計算しておく必要あり
    '''
    state_minmax = {
        'robot_position'          : (-0.43453367, 0.46022994),
        'task_space_positioin'    : (0.0, 1.0),
    }
    return state_minmax[mode]



class Sensor2Tensor_joint_space:
    def __init__(self, sensor_type):
        self.sensor_type   = sensor_type
        self.min, self.max = get_state_minmax(sensor_type)


    def get(self, db):
        robot_state  = self._get_robot_state(db["state"][self.sensor_type])
        # if robot_state.max() > self.max : self.max = robot_state.max()
        # if robot_state.min() < self.min : self.min = robot_state.min()
        robot_state = normalize(robot_state, x_min=self.min, x_max=self.max, m=0, M=1)
        assert_range(robot_state, 0, 1)
        return to_tensor(robot_state)


    def _get_robot_state(self, robot_state):
        assert_range(robot_state, self.min, self.max)
        return robot_state

