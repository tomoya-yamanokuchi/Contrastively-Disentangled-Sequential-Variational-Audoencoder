import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.visualize.SingleTensorbordSummary2PandasDataset import TensorbordSummary2PandasDataset

name  = "cdsvae4"
board = TensorbordSummary2PandasDataset(log_dir="/hdd_mount/logs_cdsvae/{}".format(name))
board.get_scalars("[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221213002108]-[remote_3090]-momo")
board.savefig("summary_fig")


board.get_scalars("[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212221821]-[melco]-neko")
board.savefig("summary_fig")