from torch import optim


class SchedulerFactory:
    def create(self, name, optimizer, max_epochs):
        assert type(name) == str

        if name == "none":
            scheduler = None

        elif name == "stepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer = optimizer,
                # step_size = max_epochs//2,
                step_size = 50, # per epoch
                gamma     = 0.5,
            )

        elif name == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer = optimizer,
                eta_min   = 2e-4,
                # T_0       = (self.config.trainer.max_epochs + 1) // 2, # originally: (opt.nEpoch+1)//2
                T_0       = (100 + 1) // 2, # originally: (opt.nEpoch+1)//2
                T_mult    = 1
            )
        else:
            raise NotImplementedError()

        return scheduler