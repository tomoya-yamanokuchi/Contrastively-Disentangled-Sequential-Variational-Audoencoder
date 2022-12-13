import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.visualize.MultipleTensorbordSummary2PandasDataset import MultipleTensorbordSummary2PandasDataset

name  = "cdsvae4"
board = MultipleTensorbordSummary2PandasDataset(log_dir="/hdd_mount/logs_cdsvae/{}".format(name))

model_list = [
    "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221213002108]-[remote_3090]-momo",
    "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212221821]-[melco]-neko",
    "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212235346]-[dl-box]-nene"
]

dataframe = board.get_scalars_as_pandas(model_list)
board.save_figure("summary_fig", dataframe)