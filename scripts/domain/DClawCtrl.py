from dataclasses import dataclass
import numpy as np

'''
・Dclaw環境に状態を与える時に使用するクラスです
・与えるべき状態のルールが記述されています
'''

@dataclass(frozen=True)
class DClawCtrl:
    task_space_abs_position : np.ndarray
    task_space_diff_position: np.ndarray
    end_effector_position   : np.ndarray
    joint_space_position    : np.ndarray
    mode                    : str = "step"

    def __post_init__(self):
        if self.mode == 'step':
            self.assert_type_shape_STEP(self.task_space_abs_position , dim=3)
            self.assert_type_shape_STEP(self.task_space_diff_position, dim=3)
            self.assert_type_shape_STEP(self.end_effector_position   , dim=9)
            self.assert_type_shape_STEP(self.joint_space_position    , dim=9)
        elif self.mode == 'sequence':
            self.assert_type_shape_SEQUENCE(self.task_space_abs_position , dim=3)
            self.assert_type_shape_SEQUENCE(self.task_space_diff_position, dim=3)
            self.assert_type_shape_SEQUENCE(self.end_effector_position   , dim=9)
            self.assert_type_shape_SEQUENCE(self.joint_space_position    , dim=9)

    def assert_type_shape_STEP(self, x, dim):
        assert type(x) == np.ndarray
        assert x.shape == (dim,)

    def assert_type_shape_SEQUENCE(self, x, dim):
        assert type(x) == np.ndarray
        assert len(x.shape) == 2 # shape = [step, dim]
        assert x.shape[-1] == dim


if __name__ == '__main__':
    import numpy as np

    state = DClawCtrl(
         task_space_abs_position = np.zeros(3),
         task_space_diff_position= np.zeros(3),
         end_effector_position   = np.zeros(9),
         joint_space_position    = np.zeros(9),
         mode = "step"
    )
    print(state.task_space_abs_position)
    print(state.end_effector_position)


    state = DClawCtrl(
         task_space_abs_position = np.zeros([5, 3]),
         task_space_diff_position= np.zeros([5, 3]),
         end_effector_position   = np.zeros([5, 9]),
         joint_space_position    = np.zeros([5, 9]),
         mode = "sequence"
    )
    print(state.task_space_abs_position)
    print(state.end_effector_position)