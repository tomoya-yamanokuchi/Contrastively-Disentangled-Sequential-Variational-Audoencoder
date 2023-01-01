from .FC_wih_Conv_FrameDecoder import FC_wih_Conv_FrameDecoder
from .FullConvFrameDecoder import FullConvFrameDecoder


class FrameDecoderFactory:
    def create(self, name, **kwargs):
        assert type(name) == str

        if   name == "fc_conv"  : return FC_wih_Conv_FrameDecoder(**kwargs)
        elif name == "full_conv": return FullConvFrameDecoder(**kwargs)
        else                    : raise NotImplementedError()