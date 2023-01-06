import os
import pandas as pd
import numpy as np
from pprint import pprint

# seabornのstyleに変更
import seaborn as sns; sns.set()
from typing import List
from typing import Dict
from pathlib import Path
from natsort import natsorted
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from .SummaryPlot import SummaryPlot


class MultipleTensorbordSummary2PandasDataset:
    WALL_TIME = 0
    STEP      = 1
    VALUE     = 2

    def y_MINMAX(self, tag):
        minmax = {
            "loss"            : (None, None),
            "Train/mse"       : (None, None),
            "Train/kld_f"     : (0, 80),
            "Train/kld_z"     : (0, 30),
            "Train/con_loss_c": (None, None),
            "Train/con_loss_m": (None, None),
            "Train/mi_fz"     : (0, 6),
        }
        if tag in minmax.keys():
            return minmax[tag]
        else:
            return (None, None)


    def __init__(self, logs: str, name: str) -> None:
        self.logs       = logs
        self.name       = name
        self.log_dir    = os.path.join(logs, name)
        self.model_list = None
        self.tags       = None


    def _get_model_list(self, search_keyward: str):
        p          = pathlib.Path(self.log_dir)
        path_list  = natsorted(list(p.glob("*")),key=lambda x:x.name)
        model_list = [str(path).split("/")[-1] for path in path_list if search_keyward in str(path)]
        return model_list


    def get_scalars_as_pandas(self, search_keyward) -> Dict:
        model_list = self._get_model_list(search_keyward)
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
            for tag in tags:
                scalars = event.Scalars(tag)
                data[model][tag] = []

                # append data
                for scalar in scalars:
                    val = scalar[MultipleTensorbordSummary2PandasDataset.VALUE]
                    # import ipdb; ipdb.set_trace()
                    data[model][tag].append(val)
        # import ipdb; ipdb.set_trace()
        return pd.DataFrame(data).T # 行と列を入れ替え


    def save_figure(self, output_dir: str, dataframe_dict: dict) -> None:
        dirname   = self._create_save_dir(output_dir)
        num_model = self._get_number_of_model(dataframe_dict)
        for tag in self.tags:
            tag_for_save = '_'.join(tag.split("/"))
            summary_plot = SummaryPlot(
                xlabel  = "step",
                ylabel  = tag_for_save,
                title   = "Number of model = {}".format(num_model),
                yminmax = self.y_MINMAX(tag)
            )
            for model_name, dataframe in dataframe_dict.items():
                print(model_name, tag)
                df = dataframe[tag]
                summary_plot.plot_mean_std(df.to_dict(), legend_label=model_name)
            summary_plot.save_fig(save_path=os.path.join(dirname, tag_for_save))


    def _create_save_dir(self, output_dir):
        dir       = os.path.join(".", output_dir)
        path_obj  = Path(os.path.join(".", output_dir))
        path_list = natsorted(list(path_obj.glob("*")),key=lambda x:x.name)
        if path_list == []:
            # version_000 がなければ作る
            number  = str(0).zfill(3)
            dirname = os.path.join(dir, "{}_version_{}".format(self.name, number))
            os.makedirs(dirname)
        else:
            # すでにあれば追加して作る
            latest_path   = str(path_list[-1])
            latent_name   = latest_path.split("/")[-1]
            latent_number = latent_name[-3:]

            print(latent_name)
            print(latent_number)
            print(int(latent_number))
            print(str(int(latent_number) + 1))
            number        = str(int(latent_number) + 1).zfill(3)
            dirname       = os.path.join(dir, "{}_version_{}".format(self.name, number))
            os.makedirs(dirname)
        # Path(os.path.join(dirname, "all")).mkdir(parents=True, exist_ok=True)
        # Path(os.path.join(dirname, "mean_std")).mkdir(parents=True, exist_ok=True)
        return dirname


    def _get_number_of_model(self, dataframe_dict: dict):
        num_model = []
        for model_name, dataframe in dataframe_dict.items():
            num_model.append(len(dataframe))
        num_model      = np.array(num_model)
        num_model_mean = int(num_model.mean())
        assert (num_model - num_model_mean).sum() == 0, print("diff = {}".format((num_model - num_model_mean)))
        return num_model_mean
