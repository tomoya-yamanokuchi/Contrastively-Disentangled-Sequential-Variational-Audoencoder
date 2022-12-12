from collections import defaultdict
from ray import tune


recursive_defaultdict = lambda: defaultdict(recursive_defaultdict)
hoge = recursive_defaultdict()
hoge["A"]["T"]["G"]["C"] = tune.choice([1, 2, 3])

print(hoge["A"]["T"]["G"]["C"])