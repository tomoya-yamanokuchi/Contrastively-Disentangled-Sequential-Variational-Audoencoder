from .contrastively_disentangled_sequential_variational_autoencoder.LitContrastivelyDisentangledSequentialVariationalAutoencoder import LitContrastivelyDisentangledSequentialVariationalAutoencoder
from .evaluator.regressor.dclaw_regressor.LitDClawRegressor import LitDClawRegressor
from .evaluator.regressor.sin_regressor.LitSinRegressor import LitSinRegressor

class ModelFactory:
    def create(self, name: str):
        prefix = name.split("_")[0]
        if  prefix == "c-dsvae"        : return LitContrastivelyDisentangledSequentialVariationalAutoencoder
        elif name == "regressor_dclaw" : return LitDClawRegressor
        elif name == "regressor_sin"   : return LitSinRegressor
        else                           : raise NotImplementedError()
