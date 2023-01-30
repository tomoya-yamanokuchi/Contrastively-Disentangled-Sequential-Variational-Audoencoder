import numpy as np
from custom import assert_range, normalize, to_tensor




class Sensor2Tensor_task_space:
    def __init__(self):
        self.name = "task_space_positioin"
        self.min  = 0.0
        self.max  = 1.0


    def get(self, db):
        robot_state = db["state"][self.name]
        assert_range(robot_state, self.min, self.max)
        robot_state = self.convert_task_space_position(robot_state)
        # import ipdb; ipdb.set_trace()
        return to_tensor(robot_state)


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

