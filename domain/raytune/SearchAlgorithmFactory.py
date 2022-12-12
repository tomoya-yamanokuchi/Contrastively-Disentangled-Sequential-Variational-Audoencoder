from ray.tune import search
from ray.tune.search import ConcurrencyLimiter

class SearchAlgorithmFactory:
    def create(self, name: str, max_concurrent: int, **kwargs):
        search_alg = search.SEARCH_ALG_IMPORT[name]()(**kwargs)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        return search_alg

