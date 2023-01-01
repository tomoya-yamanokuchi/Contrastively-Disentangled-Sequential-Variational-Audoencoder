from .variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from .contrastive_disentangled_sequential_variational_autoencoder.LitContrastiveDisentangledSequentialVariationalAutoencoder import LitContrastiveDisentangledSequentialVariationalAutoencoder

class ModelFactory:
    def create(self, name: str):
        prefix = name.split("_")[0]
        # import ipdb; ipdb.set_trace()
        if  prefix == "vae"    : return LitVariationalAutoencoder
        if  prefix == "c-dsvae": return LitContrastiveDisentangledSequentialVariationalAutoencoder
        else                   : raise NotImplementedError()
