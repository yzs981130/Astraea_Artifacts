import warnings
from abc import ABCMeta, abstractmethod


class BaseScheduler(metaclass=ABCMeta):
    def __init__(self, JOBS, CLUSTER, placement, name, logger, **kwargs):
        self.JOBS = JOBS
        self.CLUSTER = CLUSTER
        self.placement = placement
        self.name = name
        self.logger = logger
     

    @abstractmethod
    def check_resource(self, **kwargs):
        raise NotImplementedError
    

    @abstractmethod
    def place_jobs(self, **kwargs):
        raise NotImplementedError


    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError