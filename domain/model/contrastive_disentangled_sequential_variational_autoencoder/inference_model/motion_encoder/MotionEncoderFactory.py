from .FullConnectMotionEncoder import FullConnectMotionEncoder
from .LSTMMotionEncoder import LSTMMotionEncoder


class MotionEncoderFactory:
    def create(self, name, **kwargs):
        assert type(name) == str

        if   name == "fc"  : return FullConnectMotionEncoder(**kwargs)
        elif name == "lstm": return LSTMMotionEncoder(**kwargs)
        else               : raise NotImplementedError()