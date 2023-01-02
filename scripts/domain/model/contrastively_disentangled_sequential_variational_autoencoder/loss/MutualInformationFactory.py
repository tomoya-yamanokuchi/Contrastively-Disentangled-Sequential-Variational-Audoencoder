from .MutualInformation_JunwenBai import MutualInformation_JunwenBai



class MutualInformationFactory:
    def create(self, name: str, **kwargs):
        if   name == "JunwenBai": return MutualInformation_JunwenBai(**kwargs)
        else: NotImplementedError()