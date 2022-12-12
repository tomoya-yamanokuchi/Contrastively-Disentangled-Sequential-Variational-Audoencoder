import time
from pytorch_lightning.callbacks import Callback

class CallbackTrainingTime(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.time_epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        time_epoch_end = time.time()
        elapsed_time   = time_epoch_end - self.time_epoch_start
        print("         [epoch elapsed_time = {}[sec]".format(elapsed_time))

    def on_train_start(self, trainer, pl_module):
        self.time_train_start = time.time()
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        time_train_end = time.time()
        elapsed_time   = time_train_end - self.time_train_start
        print("\n\n")
        print("--------------------------------------")
        print("     total training time: ".format(elapsed_time))
        print("            sec = {:.3f}[sec]".format(elapsed_time))
        print("            min = {:.3f}[min]".format(elapsed_time / 60.))
        print("           hour = {:.3f}[h]".format(elapsed_time / 60. / 60.))
        print("--------------------------------------")