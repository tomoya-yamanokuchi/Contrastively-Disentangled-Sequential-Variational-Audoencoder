import numpy as np
import matplotlib.pyplot as plt


def save_plot_all(save_path: str, dict_summary: dict, label: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for key, val in dict_summary.items():
        ax.plot(val, label=key)

    ax.set_xlabel("steps")
    ax.set_ylabel(label)
    # ax.set_title(title, y=1.05, pad=-14)
    # ax.set_xlim(t_min, t_max)
    # ax.set_ylim(t_min, t_max)
    # ax.set_xticks([t_min, t_max])
    # ax.set_yticks([t_min, t_max])

    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc = 'upper center', ncol=1, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
    fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
    fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
    plt.close()



def save_plot_mean_std(save_path: str, dict_summary: dict, label: str):
    val_list = []
    for key, val in dict_summary.items():
        val_list.append(np.array(val))
    x = np.stack(val_list)
    num_list, num_data = x.shape

    # mu and sigma

    xx    = np.linspace(0, num_data-1, num_data)

    # plot data
    fig, ax = plt.subplots(figsize=(8, 6))

    ax_plot_mu_std(ax, *get_mu_and_bound(x))

    ax.set_xlabel("steps")
    ax.set_ylabel(label)
    ax.set_title("Number of Model = {}".format(num_list))

    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
    fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
    fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
    plt.close()


def get_mu_and_bound(x):
    mu    = np.mean(x, axis=0)
    std   = np.std(x, axis=0)
    lower = mu - 2.0*std
    upper = mu + 2.0*std
    return mu, lower, upper


def ax_plot_mu_std(ax, x, mu, lower, upper):
    ax.fill_between(x, lower, upper, alpha=0.8, label="variance", color="skyblue")
    ax.plot(x, mu, color="b", label="mean")




def plot_2D_latent_space(x, c, save_path):
    plt.figure(figsize=(13, 10))
    plt.scatter(x[:, 0], x[:, 1],
        c=c, cmap='jet',
        s=100, alpha=0.5
    )
    # plt.axis('off')
    plt.colorbar()
    plt.savefig(save_path)