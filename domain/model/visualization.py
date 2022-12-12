from torch import Tensor
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



def samples(data, label, save_path):
    if isinstance(data, Tensor):
        data = data.cpu().numpy()

    if label is not None:
        if isinstance(label, Tensor):
            label = label.cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.scatter(data[:, 0], data[:, 1], edgecolor="#333", c=label, cmap="jet")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    # plt.legend()
    plt.savefig(save_path)