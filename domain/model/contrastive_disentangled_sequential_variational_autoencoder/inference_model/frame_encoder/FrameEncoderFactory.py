from .Conv_with_FullConnect_FrameEncoder import Conv_with_FullConnect_FrameEncoder
from .FullConvFrameEncoder import FullConvFrameEncoder


class FrameEncoderFactory:
    def create(self, name, **kwargs):
        assert type(name) == str

        if   name == "conv_with_fc"  : return Conv_with_FullConnect_FrameEncoder(**kwargs)
        elif name == "full_conv"     : return FullConvFrameEncoder(**kwargs)
        else                         : raise NotImplementedError()