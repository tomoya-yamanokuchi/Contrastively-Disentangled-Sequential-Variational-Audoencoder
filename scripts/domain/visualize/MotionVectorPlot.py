import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np



class MotionVectorPlot:
    def __init__(self, title: str):
        # self.xlabel       = xlabel
        # self.ylabel       = ylabel
        self.title        = title
        # self.yminmax      = yminmax
        self.fig, self.ax = plt.subplots(figsize=(6.3, 6))
        self.fontsize1    = 15
        # self.ax.set_xlabel(xlabel, fontsize=self.fontsize1)
        # self.ax.set_ylabel(ylabel, fontsize=self.fontsize1)
        self.ax.set_title(title, fontsize=self.fontsize1)
        self.cmap = "tab10"


    def scatter_circle(self, x, color, markersize, label):
        assert len(x.shape) == 2
        self.ax.scatter(x[:, 0], x[:, 1], c=color, cmap=self.cmap, s=markersize, alpha=0.3, marker="o", label=label)

    def write_text(self, x, text, fontsize=10):
        # self.ax.text(x[:, 0], x[:, 1], text, fontsize=1, horizontalalignment="center", verticalalignment="center")
        assert len(x.shape) == 2
        for i in range(x.shape[0]):
            self.ax.text(x[i, 0], x[i, 1], text[i], fontsize=fontsize, horizontalalignment="center", verticalalignment="center")


    def scatter_cross(self, x, color, markersize, label):
        assert len(x.shape) == 2
        # assert len(c.shape) == 1
        self.ax.scatter(x[:, 0], x[:, 1], c=color, cmap=self.cmap, s=markersize, alpha=0.7, marker="x", linewidths=2, label=label)
        # plt.axis('off')


    def save_fig(self, save_path: str, dpi:int=200):
        # import ipdb; ipdb.set_trace()
        # self._set_lim()
        # self._set_legened()
        # self.fig.colorbar(self.mappable, self.ax)
        # self.fig.colorbar()
        self.fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()


    # def _set_lim(self):
    #     self.ax.set_xlim(-4*np.pi, 4*np.pi)
    #     self.ax.set_ylim(*self.yminmax)


    def _set_legened(self):
        '''
            bbox signature (left, bottom, width, height)
            with 0, 0 being the lower left corner and 1, 1 the upper right corner
        '''
        # lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        # self.fig.legend(lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
        # self.ax.legend(loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0), fontsize=self.fontsize1-1)
        self.ax.legend(bbox_to_anchor=(1, 1.03), fontsize=self.fontsize1-1)

