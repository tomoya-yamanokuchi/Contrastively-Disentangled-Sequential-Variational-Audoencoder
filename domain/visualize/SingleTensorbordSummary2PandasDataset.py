from ntpath import join
import os
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
# seabornのstyleに変更
import seaborn as sns; sns.set()
from typing import List
from typing import Dict
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class SingleTensorbordSummary2PandasDataset:
    WALL_TIME = 0
    STEP      = 1
    VALUE     = 2

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.model   = None


    def get_scalars(self, model: str) -> Dict:
        self.model = model
        path       = os.path.join(self.log_dir, model)
        files      = [path for path in Path(path).glob("**/*") if path.is_file()]
        data       = {file.parent.name: {} for file in files}

        for file in files:
            event = EventAccumulator(str(file))
            event.Reload()
            tags = event.Tags()["scalars"]
            # print(file, tags)
            for tag in tags:
                scalars = event.Scalars(tag)
                data[file.parent.name][tag] = []

                # データの格納
                for scalar in scalars:
                    data[file.parent.name][tag].append(scalar[SingleTensorbordSummary2PandasDataset.VALUE])
        self.scalars = data


    def savefig(self, output_dir: str, extention: str="png") -> None:
        path = os.path.join(self.log_dir, self.model, output_dir)
        Path(path).mkdir(parents=True, exist_ok=True)

        for file in self.scalars.keys():
            for tag in self.scalars[file].keys():

                df = pd.DataFrame(self.scalars[file][tag])
                df.plot(kind="line", title=tag, legend=False)

                tag_for_save = '_'.join(tag.split("/"))
                plt.savefig(f"{self.log_dir}/{self.model}/{output_dir}/{tag_for_save}.{extention}")
                plt.close()
