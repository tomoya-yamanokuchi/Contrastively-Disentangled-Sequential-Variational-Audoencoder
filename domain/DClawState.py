from dataclasses import dataclass
from mimetypes import init
import numpy as np

'''
・Dclaw環境に状態を与える時に使用するクラスです
・与えるべき状態のルールが記述されています
'''

@dataclass(frozen=True)
class DClawState:
    '''
    modeについて：
        永続化するときには系列になった値オブジェクトとして保存したいが，系列とステップごととで
        shapeに対するassetの掛け方が変化するようしたい．このassertの掛け方を判断するのがmode．
        - mode = "step": ステップデータとしてのshpaeをassert
        - mode = "sequence": 系列データとしてのshapeをassert
    '''
    robot_position       : np.ndarray
    object_position      : np.ndarray
    robot_velocity       : np.ndarray
    object_velocity      : np.ndarray
    end_effector_position: np.ndarray
    task_space_positioin : np.ndarray
    mode                 : str = "step"

    def __post_init__(self):
        if   self.mode == "step"    : self.__assert_step__()
        elif self.mode == "sequence": self.__assert_sequence__()
        else                        : raise NotImplementedError()

    def __assert_step__(self):
        self.assert_type_shape_dim_STEP(self.robot_position,  dim=9)
        self.assert_type_shape_dim_STEP(self.robot_velocity,  dim=9)
        self.assert_type_shape_dim_STEP(self.object_position, dim=1)
        self.assert_type_shape_dim_STEP(self.object_velocity, dim=1)

    def __assert_sequence__(self):
        self.assert_type_shape_dim_SEQUENCE(self.robot_position,  dim=9)
        self.assert_type_shape_dim_SEQUENCE(self.robot_velocity,  dim=9)
        self.assert_type_shape_dim_SEQUENCE(self.object_position, dim=1)
        self.assert_type_shape_dim_SEQUENCE(self.object_velocity, dim=1)


    def assert_type_shape_dim_STEP(self, x, dim):
        val_type = type(x)
        if   val_type == np.ndarray             :   assert x.shape == (dim,)
        elif (dim == 1) and (val_type == float) :   pass
        else                                    :   raise NotImplementedError()

    def assert_type_shape_dim_SEQUENCE(self, x, dim):
        assert      type(x) == np.ndarray
        assert len(x.shape) == 2  # shape = [step, dim]
        assert  x.shape[-1] == dim


if __name__ == '__main__':
    import numpy as np

    state = DClawState(
        robot_position        = np.zeros(9),
        object_position       = np.zeros(1),
        robot_velocity        = np.zeros(9),
        object_velocity       = np.zeros(1),
        # force                 = np.random.randn(9),
        end_effector_position = np.zeros(9),
        task_space_positioin  = np.zeros(3),
        mode="step"
    )
    print(state.robot_position)
    print(state.object_position)
    print(state.mode)

    step = 8
    state = DClawState(
        robot_position        = np.zeros([step, 9]),
        object_position       = np.zeros([step, 1]),
        robot_velocity        = np.zeros([step, 9]),
        object_velocity       = np.zeros([step, 1]),
        end_effector_position = np.zeros([step, 9]),
        task_space_positioin  = np.zeros([step, 3]),
        mode="sequence",
    )
    print(state.robot_position)
    print(state.object_position)