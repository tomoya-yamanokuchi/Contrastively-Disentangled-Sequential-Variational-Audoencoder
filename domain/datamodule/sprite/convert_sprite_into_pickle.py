from tqdm import *
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim
import cv2
import numpy as np

class Sprites(torch.utils.data.Dataset):
    def __init__(self,path,size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return torch.load(self.path+'/%d.sprite' % (idx+1))



sprite     = Sprites('data/Sprite/lpc-dataset/train', 100)
batch_size = 25
loader     = torch.utils.data.DataLoader(sprite, batch_size=batch_size, shuffle=True, num_workers=4)
            MNIST(self.data_dir, train=False, transform=self.transform)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# for epoch in range(0, loader.dataset.length):
#     print("Running Epoch : {}".format(epoch+1))


for i,dataitem in tqdm(enumerate(loader, 1)):
    # _,_,_,_,_,_,data = dataitem
    print("loader : ", i)

    images = []
    for k, d in enumerate(dataitem):
        print("dataitem : ", k)
        d = np.array(d) # (step, channel, w, h)

        # for t in range(d.shape[0]):
        for t in range(1):
            dt = d[t] # (channel, w, h)
            dt = np.transpose(dt, (1, 2, 0))
            cv2.imshow("img", dt)
            cv2.waitKey(50)
