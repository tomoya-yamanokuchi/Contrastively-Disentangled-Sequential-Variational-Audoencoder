from ntpath import join
import os
from unicodedata import name
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
# seabornのstyleに変更
import seaborn as sns; sns.set()
from typing import List
from typing import Dict
from pathlib import Path
from natsort import natsorted
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class MultipleTensorbordSummary2PandasDataset:
    WALL_TIME = 0
    STEP      = 1
    VALUE     = 2

    def __init__(self, log_dir: str) -> None:
        self.log_dir    = log_dir
        self.model_list = None
        self.tags       = None


    def get_scalars_as_pandas(self, model_list: List[str]) -> Dict:
        assert type(model_list) is list
        self.model_list = model_list

        # create initial data dictionary
        data = {}
        for model in model_list:
            assert type(model) is str
            data[model] = {}

        # get scalars from each model
        for model in model_list:
            path       = os.path.join(self.log_dir, model)
            event_file = [path for path in Path(path).glob("**/*") if 'events.out' in str(path)]
            assert len(event_file) == 1
            event_file = event_file[0]
            event      = EventAccumulator(str(event_file)).Reload()
            tags       = event.Tags()["scalars"]
            if self.tags is None:
                self.tags = tags
            for tag in tags[3:]:
                scalars = event.Scalars(tag)
                data[model][tag] = []

                # append data
                for scalar in scalars:
                    val = scalar[MultipleTensorbordSummary2PandasDataset.VALUE]
                    # import ipdb; ipdb.set_trace()
                    data[model][tag].append(val)
        # import ipdb; ipdb.set_trace()
        return pd.DataFrame(data).T # 行と列を入れ替え


    def save_figure(self, output_dir: str, dataframe: pd.DataFrame, extention: str="png") -> None:
        assert isinstance(output_dir, str)
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(extention, str)

        dir       = os.path.join(".", output_dir)
        path_obj  = Path(os.path.join(".", output_dir))
        path_list = natsorted(list(path_obj.glob("*")),key=lambda x:x.name)
        if path_list == []:
            # version_000 がなければ作る
            number  = str(0).zfill(3)
            dirname = os.path.join(dir, "version_" + number)
            os.makedirs(dirname)
        else:
            # すでにあれば追加して作る
            latest_path   = str(path_list[-1])
            latent_name   = latest_path.split("/")[-1]
            latent_number = latent_name[-3:]
            number        = str(int(latent_number) + 1).zfill(3)
            dirname       = os.path.join(dir, "version_" + number)
            os.makedirs(dirname)

        Path(os.path.join(dirname, "all")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dirname, "mean_std")).mkdir(parents=True, exist_ok=True)

        for tag in self.tags[3:]:
            df           = dataframe[tag]
            dict_summary = df.to_dict()
            tag_for_save = '_'.join(tag.split("/"))

            # plot_func[mode](save_path, dict_summary, label=tag)
            self._save_plot_all(     save_path=f"{dirname}/all/{tag_for_save}.{extention}", dict_summary=dict_summary, label=tag)
            self._save_plot_mean_std(save_path=f"{dirname}/mean_std/{tag_for_save}.{extention}", dict_summary=dict_summary, label=tag)


    def _save_plot_all(self, save_path: str, dict_summary: dict, label: str):
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

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'upper center', ncol=1, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
        fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        plt.close()


    def _save_plot_mean_std(self, save_path: str, dict_summary: dict, label: str):
        val_list = []
        for key, val in dict_summary.items():
            val_list.append(np.array(val))
        x = np.stack(val_list)
        num_list, num_data = x.shape

        # mu and sigma
        mu    = np.mean(x, axis=0)
        std   = np.std(x, axis=0)
        lower = mu - 2.0*std
        upper = mu + 2.0*std
        xx    = np.linspace(0, num_data-1, num_data)

        # plot data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.fill_between(xx, lower, upper, alpha=0.8, label="variance", color="skyblue")
        ax.plot(xx, mu, color="b", label="mean")
        # ax.plot(val, label=key)

        ax.set_xlabel("steps")
        ax.set_ylabel(label)
        ax.set_title("Number of Model = {}".format(num_list))

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0, 0.6, 0.9, 0.45), fontsize=10)
        fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        fig.savefig(save_path, bbox_inches='tight') #, pad_inches=0.1)
        plt.close()