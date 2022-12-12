from PIL import Image
import glob
from natsort import natsorted

pathname = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_205/reconstruction"
files    = natsorted(glob.glob(pathname + '/*.png'))[:100][::5]
# import ipdb; ipdb.set_trace()
images   = list(map(lambda file : Image.open(file) , files))

# import ipdb; ipdb.set_trace()
images[0].save(pathname + '/image.gif' , save_all = True , append_images = images[1:] , duration = 100 , loop = 0)