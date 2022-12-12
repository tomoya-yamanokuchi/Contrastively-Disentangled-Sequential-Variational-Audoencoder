from ray.tune import schedulers

class SchedulerFactory:
    def create(self, name: str, **kwargs):
        return schedulers.SCHEDULER_IMPORT[name](**kwargs)
