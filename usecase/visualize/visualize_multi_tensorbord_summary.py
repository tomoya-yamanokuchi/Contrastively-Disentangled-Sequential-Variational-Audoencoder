from pprint import pprint
from natsort import natsorted
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.visualize.MultipleTensorbordSummary2PandasDataset import MultipleTensorbordSummary2PandasDataset

# name  = "cdsvae4"
# name  = "cdsvae_datamodule_sprite_JunwenBi"
name  = "cdsvae_action_norm_valve"
board = MultipleTensorbordSummary2PandasDataset(logs="/hdd_mount/logs_cdsvae/", name=name)

mode = True
if mode:
    # model_list = [
    #     "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221213002108]-[remote_3090]-momo",
    #     "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212221821]-[melco]-neko",
    #     "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212235346]-[dl-box]-nene"
    # ]

    # model_list = [
    #     "[c-dsvae]-[sprite_JunwenBi]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221221072950]-[remote_3090]-",
    #     "[c-dsvae]-[sprite_JunwenBi]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221221072930]-[melco]-",
    # ]

    model_list = [
        '[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[500epoch]-[20221221152837]-[remote_3090]-kkk',
        '[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[500epoch]-[20221221174417]-[remote_3090]-kkk',
        '[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[500epoch]-[20221221200008]-[remote_3090]-kkk',
        '[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[500epoch]-[20221221221550]-[remote_3090]-kkk',
        '[c-dsvae]-[action_norm_valve]-[dim_f=256]-[dim_z=32]-[500epoch]-[20221222003141]-[remote_3090]-kkk',
    ]

else:
    p          = pathlib.Path(board.log_dir)
    path_list  = natsorted(list(p.glob("*")),key=lambda x:x.name)
    model_list = [str(path).split("/")[-1] for path in path_list]

pprint(model_list)

dataframe = board.get_scalars_as_pandas(model_list)
board.save_figure("summary_fig", dataframe)