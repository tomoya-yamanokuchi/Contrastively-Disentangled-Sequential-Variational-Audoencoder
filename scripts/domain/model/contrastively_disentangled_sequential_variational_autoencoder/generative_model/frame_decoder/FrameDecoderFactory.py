from .FullConvFrameDecoder import FullConvFrameDecoder


class FrameDecoderFactory:
    def create(self, name, **kwargs):
        assert type(name) == str
        if name == "full_conv": return FullConvFrameDecoder(**kwargs)
        else                  : raise NotImplementedError()