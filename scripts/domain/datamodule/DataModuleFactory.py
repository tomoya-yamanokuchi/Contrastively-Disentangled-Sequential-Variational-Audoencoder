from .sprite_JunwenBai.SpriteJunwenBaiDataModule import SpriteJunwenBaiDataModule


class DataModuleFactory:
    def create(self, name: str, **kwargs):
        if name == "sprite_JunwenBi"  : return SpriteJunwenBaiDataModule(**kwargs)
        else: NotImplementedError()

