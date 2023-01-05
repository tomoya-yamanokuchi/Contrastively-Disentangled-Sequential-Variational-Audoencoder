from .MutualInformation_JunwenBai import MutualInformation_JunwenBai
from .MutualInformation_myfunc import MutualInformation_myfunc


class MutualInformationFactory:
    def create(self, name: str, **kwargs):
        if    name == "JunwenBai": return MutualInformation_JunwenBai(**kwargs)
        elif  name == "mylogdensity": return MutualInformation_myfunc(**kwargs)
        else: NotImplementedError()