import numpy as np
import matplotlib.pyplot as plt



class SummaryPlot:
    def __init__(self, xlabel: str, ylabel: str, title: str, yminmax=tuple):
        self.xlabel       = xlabel
        self.ylabel       = ylabel
        self.title        = title
        self.yminmax      = yminmax
        self.fig, self.ax = plt.subplots(figsize=(5.5, 4))
        self.fontsize1    = 15
        self.ax.set_xlabel(xlabel, fontsize=self.fontsize1)
        self.ax.set_ylabel(ylabel, fontsize=self.fontsize1)
        self.ax.set_title(title, fontsize=self.fontsize1)


    def plot_mean_std(self, dict_summary: dict, legend_label: str):
        val_list = []
        for key, val in dict_summary.items():
            val_list.append(np.array(val))
        y = np.stack(val_list)
        _, num_data   = y.shape
        self.num_data = num_data
        x = np.linspace(0, num_data-1, num_data)
        self._plot_mu_std(x, *self._get_mu_and_bound(y), legend_label)


    def _plot_mu_std(self, x, mu, lower, upper, legend_label):
        self.ax.fill_between(x, lower, upper, alpha=0.3)
        self.ax.plot(x, mu, label=legend_label)


    def _get_mu_and_bound(self, x):
        assert len(x.shape) == 2 # (num_model, num_step)
        mu    = np.mean(x, axis=0)
        std   = np.std(x, axis=0)
        lower = mu - 2.0*std
        upper = mu + 2.0*std
        return mu, lower, upper


    def save_fig(self, save_path: str):
        # import ipdb; ipdb.set_trace()
        self._set_lim()
        self._set_legened()
        self.fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        plt.close()


    def _set_lim(self):
        self.ax.set_xlim(0, self.num_data)
        self.ax.set_ylim(*self.yminmax)


    def _set_legened(self):
        '''
            bbox signature (left, bottom, width, height)
            with 0, 0 being the lower left corner and 1, 1 the upper right corner
        '''
        # lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        # self.fig.legend(lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
        self.ax.legend(loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0), fontsize=self.fontsize1-1)
