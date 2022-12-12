from .mnist.MNISTDataModule import MNISTDataModule
from .sprite.SpriteDataModule import SpriteDataModule
from .sprite_augmentation.SpriteAugmentationDataModule import SpriteAugmentationDataModule
from .sprite_JunwenBai.SpriteJunwenBaiDataModule import SpriteJunwenBaiDataModule
from .robel_claw_valve.ActionNormalizedValveDataModule import ActionNormalizedValveDataModule


'''
データローダー直す
'''

class DataModuleFactory:
    def create(self, name: str, **kwargs: str):
        name = name.lower()
        if   name == "mnist"            : return MNISTDataModule(**kwargs)
        elif name == "sprite"           : return SpriteDataModule(**kwargs)
        elif name == "sprite_aug"       : return SpriteAugmentationDataModule(**kwargs)
        elif name == "sprite_jb"        : return SpriteJunwenBaiDataModule(**kwargs)
        elif name == "action_norm_valve": return ActionNormalizedValveDataModule(**kwargs)
        else: NotImplementedError()