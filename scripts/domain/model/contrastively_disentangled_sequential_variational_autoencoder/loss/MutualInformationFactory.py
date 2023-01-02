from .MutualInformation_JunwenBi import MutualInformation_JunwenBi



class MutualInformationFactory:
    def create(self, name: str, **kwargs):
        if   name == "JunwenBi": return MutualInformation_JunwenBi(**kwargs)
        else: NotImplementedError()