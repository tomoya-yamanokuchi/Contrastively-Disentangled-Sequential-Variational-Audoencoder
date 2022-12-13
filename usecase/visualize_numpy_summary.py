import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import os
import numpy as np


model    = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212180321]-[melco]-"
# ----------------------------------------------------------------------------------
name     = "cdsvae2"
save_dir = "/hdd_mount/logs_cdsvae/{}/".format(name)
log_dir  = save_dir + model
os.makedirs(log_dir + "/fig_summary", exist_ok=True)

summary = np.load(log_dir + "/numpy_summary.npy", allow_pickle=True)


summ = summary.all()


for key, val in summ.items():
    plt.plot(val)
    # plt.show()
    plt.savefig(log_dir + "/fig_summary" + "/{}.png".format(key))