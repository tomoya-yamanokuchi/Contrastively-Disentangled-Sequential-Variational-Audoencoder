import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from custom.visualize.VectorHeatmap import VectorHeatmap



vector_heatmap = VectorHeatmap()


a = np.random.randn(5, 5)

vector_heatmap.pause_show(a, interval=10)
