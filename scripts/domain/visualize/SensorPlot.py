import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np

"""
def sin_data_plot(x, y, x_true, y_true, file_path):
    plt.clf()
    # import ipdb; ipdb.set_trace()
    plt.plot(x, y, "x", color="g", markersize=5, label="observation data")
    plt.plot(x_true, y_true, "-", color="k", label="true function")
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y=sin(x)", fontsize=15)
    plt.title("N = {}".format(str(x.shape[0])))
    plt.legend(loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0), fontsize=8)
    plt.savefig(file_path)
"""


class SensorPlot:
    def __init__(self, xlabel: str, title: str):
        self.xlabel            = xlabel
        # self.ylabel            = ylabel
        self.title             = title
        self.yminmax           = (-0.1, 1.1)
        self.fontsize1         = 15
        self.x_max             = -1e4
        self.x_min             = 1e4
        # << plot related parameters >>
        self.dim               = 6 # (cos, sin -> task_space_postion x3) + (cos, sin -> valve_angle)
        self.fig, self.ax      = plt.subplots(nrows=self.dim, ncols=1, figsize=(5, 12))
        # self.nrows, self.ncols = self.ax.shape
        [ax.set_xlabel(xlabel, fontsize=self.fontsize1) for ax in self.ax.reshape(-1)]
        # [ax.set_ylabel(ylabel, fontsize=self.fontsize1) for ax in self.ax.reshape(-1)]
        [ax.set_ylabel("dim{}".format(i), fontsize=self.fontsize1) for i, ax in enumerate(self.ax.reshape(-1))]
        # self.ax.set_title(title, fontsize=self.fontsize1)
        # import ipdb; ipdb.set_trace()


    def update_xminmax(self, x):
        x_min = x.min()
        x_max = x.max()
        if x_min < self.x_min: self.x_min = x_min
        if x_max > self.x_max: self.x_max = x_max


    def plot(self, x, y):
        assert len(x.shape) == 1
        assert len(y.shape) == 2
        assert  y.shape[-1] == self.dim
        self.update_xminmax(x)
        [ax.plot(x, yd.squeeze(-1), "-", label="true function") for ax, yd in zip(self.ax.reshape(-1), np.split(y, self.dim, axis=-1))]


    def plot_true(self, x, y):
        assert len(x.shape) == 1
        assert len(y.shape) == 2
        assert  y.shape[-1] == self.dim
        self.update_xminmax(x)
        [ax.plot(x, yd.squeeze(-1), "--", color="red", label="true function") for ax, yd in zip(self.ax.reshape(-1), np.split(y, self.dim, axis=-1))]


    def save_fig(self, save_path: str):
        # import ipdb; ipdb.set_trace()
        self.ax = np.transpose(self.ax)
        self._set_lim()
        self._set_legened()
        self.fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        plt.close()


    def _set_lim(self):
        for ax in self.ax.reshape(-1):
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(*self.yminmax)


    def _set_legened(self):
        '''
            bbox signature (left, bottom, width, height)
            with 0, 0 being the lower left corner and 1, 1 the upper right corner
        '''
        # lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        # self.fig.legend(lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
        # [self.ax[d].legend(loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0), fontsize=self.fontsize1-1) for d in range(self.dim)]
