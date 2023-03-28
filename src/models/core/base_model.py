import os
from abc import abstractmethod
from functools import partial
import collections
import torch
# import torch.nn as nn


import core.util as Util
CustomResult = collections.namedtuple('CustomResult', 'name result')

# TODO: Add normal paths
checkpoint_path: str
resume_path: str
phase: str
batch_size: int
max_epochs: int
max_steps: int
save_every_n_checkpoints: int
val_every_n_checkpoints: int

# which GPU training is on
global_rank: int


class BaseModel():
    def __init__(self, opt, phase_loader, val_loader, metrics):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.opt = opt
        self.phase = phase
        self.set_device = partial(Util.set_device, rank=opt['global_rank'])

        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = batch_size
        self.epoch = 0
        self.step = 0

        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics

        self.results_dict = CustomResult([], [])  # {"name":[], "result":[]}

    def train(self):
        while self.epoch <= max_epochs and self.step <= max_steps:
            self.epoch += 1

            self.train_step()

            if self.epoch % save_every_n_checkpoints == 0:
                self.save_everything()

            if self.epoch % val_every_n_checkpoints == 0:
                if self.val_loader is None:
                    pass
                else:
                    self.val_step()

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError(
            'You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError(
            'You must specify how to do validation on your networks.')

    def test_step(self):
        pass

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        if global_rank != 0:
            return
        
        save_filename = f'{self.epoch}_{network_label}.pth'
        save_path = os.path.join(checkpoint_path, save_filename)

        # if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        #     network = network.module

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if resume_path is None:
            return

        model_path = f"{resume_path}_{network_label}.pth"

        if not os.path.exists(model_path):
            return

        # if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        #     network = network.module

        network.load_state_dict(
            torch.load(
                model_path,
                map_location=lambda storage,
                loc: Util.set_device(storage)
            ),
            strict=strict
        )

    # def save_training_state(self):
    #     """ saves training state during training, only work on GPU 0 """
    #     if global_rank != 0:
    #         return

    #     assert isinstance(self.optimizers, list) and isinstance(
    #         self.schedulers, list), 'optimizers and schedulers must be a list.'

    #     state = {'epoch': self.epoch, 'step': self.step,
    #              'schedulers': [], 'optimizers': []}

    #     for s in self.schedulers:
    #         state['schedulers'].append(s.state_dict())

    #     for o in self.optimizers:
    #         state['optimizers'].append(o.state_dict())

    #     save_filename = f'{self.epoch}.state'
    #     save_path = os.path.join(checkpoint_path, save_filename)
    #     torch.save(state, save_path)

    # def resume_training(self):
    #     """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
    #     if self.phase != 'train' or resume_path is None:
    #         return

    #     assert isinstance(self.optimizers, list) and isinstance(
    #         self.schedulers, list), 'optimizers and schedulers must be a list.'

    #     state_path = f"{resume_path}.state"

    #     if not os.path.exists(state_path):
    #         return

    #     resume_state = torch.load(
    #         state_path,
    #         map_location=lambda storage,
    #         loc: self.set_device(storage)
    #     )

    #     resume_optimizers = resume_state['optimizers']
    #     resume_schedulers = resume_state['schedulers']

    #     assert len(resume_optimizers) == len(self.optimizers), f'Wrong lengths of optimizers {len(resume_optimizers)} != {len(self.optimizers)}'
    #     assert len(resume_schedulers) == len(self.schedulers), f'Wrong lengths of schedulers {len(resume_schedulers)} != {len(self.schedulers)}'

    #     for i, o in enumerate(resume_optimizers):
    #         self.optimizers[i].load_state_dict(o)

    #     for i, s in enumerate(resume_schedulers):
    #         self.schedulers[i].load_state_dict(s)

    #     self.epoch = resume_state['epoch']

    #     self.step = resume_state['step']

    # def load_everything(self):
    #     pass

    @abstractmethod
    def save_everything(self):
        raise NotImplementedError(
            'You must specify how to save your networks, optimizers and schedulers.')
