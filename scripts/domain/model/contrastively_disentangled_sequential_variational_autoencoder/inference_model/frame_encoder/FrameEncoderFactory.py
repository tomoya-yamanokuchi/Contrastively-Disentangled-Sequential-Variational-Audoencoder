from .FullConvFrameEncoder import FullConvFrameEncoder


class FrameEncoderFactory:
    def create(self, name, **kwargs):
        assert type(name) == str
        if name == "full_conv" : return FullConvFrameEncoder(**kwargs)
        else                   : raise NotImplementedError()