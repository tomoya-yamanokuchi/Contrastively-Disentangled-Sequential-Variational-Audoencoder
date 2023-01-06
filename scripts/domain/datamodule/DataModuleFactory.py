from .sprite_JunwenBai.SpriteJunwenBaiDataModule import SpriteJunwenBaiDataModule
from .robel_claw_valve.ActionNormalizedValveDataModule import ActionNormalizedValveDataModule

class DataModuleFactory:
    def create(self, name: str, **kwargs):
        if   name == "sprite_JunwenBai" : return SpriteJunwenBaiDataModule(**kwargs)
        elif name == "action_norm_valve": return ActionNormalizedValveDataModule(**kwargs)
        else: NotImplementedError()

