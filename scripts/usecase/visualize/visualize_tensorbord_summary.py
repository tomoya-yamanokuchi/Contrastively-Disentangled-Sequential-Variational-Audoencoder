from pprint import pprint
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.visualize.MultipleTensorbordSummary2PandasDataset import MultipleTensorbordSummary2PandasDataset


# board            = MultipleTensorbordSummary2PandasDataset(logs="/hdd_mount/logs_cdsvae/", name="cdsvae_sprite")
# dataframe_cdsvae = board.get_scalars_as_pandas(search_keyward="melco")
# dataframe_naive  = board.get_scalars_as_pandas(search_keyward="remote3090")

# dataframe_cdsvae = board.get_scalars_as_pandas(search_keyward="new_logdensity_cdsvae")
# dataframe_naive  = board.get_scalars_as_pandas(search_keyward="new_logdensity_naive_dsvae")



board            = MultipleTensorbordSummary2PandasDataset(logs="/hdd_mount/logs_cdsvae/", name="cdsvae_dclaw")
dataframe_cdsvae = board.get_scalars_as_pandas(search_keyward="melco")
dataframe_naive  = board.get_scalars_as_pandas(search_keyward="remote3090")

board.save_figure(
    output_dir = "summary_fig",
    dataframe_dict = {
        "DSVAE"  : dataframe_naive,
        "C-DSVAE": dataframe_cdsvae,
    }
)