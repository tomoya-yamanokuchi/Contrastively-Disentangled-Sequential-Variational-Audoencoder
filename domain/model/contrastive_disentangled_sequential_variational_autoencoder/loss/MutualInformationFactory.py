from .MutualInformation_JunwenBi import MutualInformation_JunwenBi
from .MutualInformation_develop  import MutualInformation_develop


class MutualInformationFactory:
    def create(self, name: str, **kwargs):
        # name = name.lower()
        # import ipdb; ipdb.set_trace()

        if   name == "JunwenBi": return MutualInformation_JunwenBi(**kwargs)
        elif name == "develop" : return MutualInformation_develop(**kwargs)
        else: NotImplementedError()