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


class SinDataPlot:
    def __init__(self, xlabel: str, ylabel: str, title: str, yminmax=tuple):
        self.xlabel       = xlabel
        self.ylabel       = ylabel
        self.title        = title
        self.yminmax      = yminmax
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fontsize1    = 15
        self.ax.set_xlabel(xlabel, fontsize=self.fontsize1)
        self.ax.set_ylabel(ylabel, fontsize=self.fontsize1)
        self.ax.set_title(title, fontsize=self.fontsize1)


    def plot_prediction(self, x, mean, std):
        assert len(x.shape) == 1
        assert len(mean.shape) == 1
        assert len(std.shape) == 1
        lower = mean - 2.0*std
        upper = mean + 2.0*std
        self.ax.fill_between(x, lower, upper, alpha=0.3)
        self.ax.plot(x, mean, label="prediction")


    def plot_observation(self, x, y):
        self.ax.plot(x, y, "x", color="g", markersize=5, label="observation data")


    def plot_true_function(self, x, y):
        self.ax.plot(x, y, "-", color="k", label="true function")


    def save_fig(self, save_path: str):
        # import ipdb; ipdb.set_trace()
        self._set_lim()
        self._set_legened()
        self.fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        plt.close()


    def _set_lim(self):
        self.ax.set_xlim(-4*np.pi, 4*np.pi)
        self.ax.set_ylim(*self.yminmax)


    def _set_legened(self):
        '''
            bbox signature (left, bottom, width, height)
            with 0, 0 being the lower left corner and 1, 1 the upper right corner
        '''
        # lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        # self.fig.legend(lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
        self.ax.legend(loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0), fontsize=self.fontsize1-1)
