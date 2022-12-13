import pandas as pd
import matplotlib.pyplot as plt

# seabornのstyleに変更
import seaborn as sns; sns.set()

from typing import Dict
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorbordSummaryVisualization:
    WALL_TIME = 0
    STEP = 1
    VALUE = 2

    def __init__(self, log_dir: str) -> None:
        self.log_dir   = log_dir
        self.log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]
        self.scalars   = self.get_scalars()
        # self._logger = get_logger()

    def get_scalars(self) -> Dict:
        data = {log_file.parent.name: {} for log_file in self.log_files}

        for log_file in self.log_files:
            event = EventAccumulator(str(log_file))
            event.Reload()

            # tags == ["loss", "accuracy", "lr"]
            tags = event.Tags()["scalars"]

            for tag in tags:
                scalars = event.Scalars(tag)
                data[log_file.parent.name][tag] = []

                # データの格納
                for scalar in scalars:
                    data[log_file.parent.name][tag].append(scalar[TensorbordSummaryVisualization.VALUE])

        return data

    def savefig(self, output_dir: str, extention: str="png") -> None:
        Path(f"{self.log_dir}/{output_dir}").mkdir(parents=True, exist_ok=True)
        import ipdb; ipdb.set_trace()

        for file in self.scalars.keys():
            for tag in self.scalars[file].keys():

                # import ipdb; ipdb.set_trace()
                df = pd.DataFrame(self.scalars[file][tag])
                df.plot(kind="line", title=tag, legend=False)

                tag_for_save = tag.split("/")[-1]
                plt.savefig(f"{self.log_dir}/{output_dir}/{tag_for_save}.{extention}")
                plt.close()
