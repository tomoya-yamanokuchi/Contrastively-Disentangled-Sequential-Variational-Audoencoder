from .sprite_JunwenBai.SpriteJunwenBaiDataModule import SpriteJunwenBaiDataModule


class DataModuleFactory:
    def create(self, name: str, **kwargs):
        if name == "sprite_JunwenBai"  : return SpriteJunwenBaiDataModule(**kwargs)
        else: NotImplementedError()

