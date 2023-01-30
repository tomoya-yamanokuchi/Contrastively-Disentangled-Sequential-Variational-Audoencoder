from .Sensor2Tensor_joint_space import Sensor2Tensor_joint_space
from .Sensor2Tensor_task_space import Sensor2Tensor_task_space


class Sensor2TensorFactory:
    def create(self, name: str, **kwargs):
        # import ipdb; ipdb.set_trace()
        if   name == "robot_position"       : return Sensor2Tensor_joint_space()
        elif name == "task_space_positioin" : return Sensor2Tensor_task_space()
        else: NotImplementedError()

