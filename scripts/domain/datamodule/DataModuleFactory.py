from .sprite_JunwenBai.SpriteJunwenBaiDataModule import SpriteJunwenBaiDataModule
from .robel_dclaw_valve.ActionNormalizedValveDataModule import ActionNormalizedValveDataModule
from .robel_dclaw_valve_deterministic.ROBELDClawValveDeterministicDataModule import ROBELDClawValveDeterministicDataModule

'''
command for sending dataset:
    scp -r /home/tomoya-y/workspace/dataset/xxx tomoya-y@xxx.xxx.xxx.xxx:/home/tomoya-y/workspace/dataset/
'''

class DataModuleFactory:
    def create(self, name: str, **kwargs):
        if   name == "sprite_JunwenBai"          : return SpriteJunwenBaiDataModule(**kwargs)
        elif name == "action_norm_valve"         : return ActionNormalizedValveDataModule(**kwargs)
        elif name == "robel_dclaw_deterministic" : return ROBELDClawValveDeterministicDataModule(**kwargs)
        else: NotImplementedError()

