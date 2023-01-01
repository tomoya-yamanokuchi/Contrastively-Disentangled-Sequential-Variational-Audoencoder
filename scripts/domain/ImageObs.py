from dataclasses import dataclass
import numpy as np
'''
・Dclaw環境に状態を与える時に使用するクラスです
・与えるべき状態のルールが記述されています
'''

@dataclass(frozen=True)
class ImageObs:
    canonical    : np.ndarray
    random_nonfix: np.ndarray
    mode         : str = "step"

    def __post_init__(self):
        if self.mode == 'step':
            self.assert_type_shape_STEP(self.canonical , dim=3)
            self.assert_type_shape_STEP(self.random_nonfix, dim=3)
        elif self.mode == 'sequence':
            self.assert_type_shape_SEQUENCE(self.canonical , dim=3)
            self.assert_type_shape_SEQUENCE(self.random_nonfix, dim=3)

    def assert_type_shape_STEP(self, x, dim):
        # assert isinstance(x, ImageObject)
        # x = x.channel_last
        assert type(x) == np.ndarray
        assert len(x.shape) == 3
        assert x.shape[-1] == 3
        assert x.shape[0] == x.shape[1]

    def assert_type_shape_SEQUENCE(self, x, dim):
        # import ipdb; ipdb.set_trace()
        assert type(x) == np.ndarray
        assert len(x.shape) == 4 # shape = [step, dim]
        assert x.shape[-1] == 3
        assert x.shape[1] == x.shape[2]


if __name__ == '__main__':
    import numpy as np

    img = ImageObs(
        canonical     = np.zeros([64, 64, 3]),
        random_nonfix = np.zeros([64, 64, 3]),
        mode          = "step"
    )
    print(img.canonical.shape)

    img = ImageObs(
        canonical     = np.zeros([25, 62, 64, 3]),
        random_nonfix = np.zeros([25, 64, 64, 3]),
        mode          = "sequence"
    )
    print(img.canonical.shape)