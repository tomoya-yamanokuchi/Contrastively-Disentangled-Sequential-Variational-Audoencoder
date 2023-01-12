import torch
from torch import Tensor
from torchvision import utils


def save_image(save_path: str, image: Tensor):
    num_batch, step, channel, width, height = image.shape
    image_list = []
    for i in range(num_batch):
        image_list.append(utils.make_grid(image[i], nrow=step, padding=2, pad_value=0.0, normalize=True))

    # print("\n\n---------------------------------------")
    # print(" [  images ] min. max = [{}, {}]".format(   image[1].min(),    image[1].max()))
    # print("---------------------------------------\n\n")
    '''
        Plese check if range of img is [0.0, 1.0].
        Because utils.save_image() assums that tensor image is in range [0.0, 1.0] internally.
    '''
    utils.save_image(torch.cat(image_list, dim=1), fp=save_path)