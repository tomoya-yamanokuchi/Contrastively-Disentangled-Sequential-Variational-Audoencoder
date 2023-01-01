from .contrastively_disentangled_sequential_variational_autoencoder.LitContrastivelyDisentangledSequentialVariationalAutoencoder import LitContrastivelyDisentangledSequentialVariationalAutoencoder


class ModelFactory:
    def create(self, name: str):
        prefix = name.split("_")[0]
        if  prefix == "c-dsvae": return LitContrastivelyDisentangledSequentialVariationalAutoencoder
        else                   : raise NotImplementedError()
