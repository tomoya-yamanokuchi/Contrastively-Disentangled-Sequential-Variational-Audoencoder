import torch


def torch2numpy(image):
    len_shape = len(image.shape)
    if len_shape == 3:
        return image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    elif len_shape == 4:
        return image.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    else:
        raise NotImplementedError()