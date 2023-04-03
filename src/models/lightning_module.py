import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as data
import utils
from palette.Palette import Palette
import os
import copy
import collections
import os

# I want to kill myself
import sys
sys.path.insert(0, './src/data')
from make_dataset import ArtifactDataset


CustomResult = collections.namedtuple('CustomResult', 'name result')

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Diffusar(pl.LightningModule):
    def __init__(self, unet_name, unet_args, scheduler_args, ema_scheduler):
        super().__init__()
        
        self.network = utils.get_network(unet_name, unet_args, scheduler_args)
        self.loss_fn = utils.get_loss_fn()

        self.network.set_loss(self.loss_fn)
        self.network.set_new_noise_schedule(device=self.device)

        self.network.train()

        self.step = 0

        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.network_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
            self.network_EMA = self.network_EMA.to(self.device)

    def setup():
        pass

    def configure_optimizers(self):
        optimizer = utils.get_optimizer(self.network, self.optimizer_name, self.optimizer_args)
        return optimizer

    # def forward(self, x):
    #     # x = torch.relu(self.fc1(x))
    #     # x = self.fc2(x)
    #     # return x
    #     pass
        
    def training_step(self, batch, batch_idx):
        self.step += 1
        # self.set_input(batch)
        # self.source_image = batch.get('source_image').to(self.device)
        # self.target_image = batch.get('target_image').to(self.device)

        self.source_image = batch['source_image']
        self.target_image = batch['target_image']
        
        loss = self.network(self.target_image, self.source_image)

        if self.step > self.ema_scheduler['ema_start'] and self.step % self.ema_scheduler['ema_step'] == 0:
            self.EMA.update_model_average(self.network_EMA, self.network)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = nn.functional.mse_loss(y_hat, y)
        # self.log('val_loss', loss)
        pass

    def on_save_checkpoint(self, checkpoint):
        # checkpoint['my_additional_info'] = "Some additional info to be saved"
        # return checkpoint
        pass

    def on_load_checkpoint(self, checkpoint):
        # self.my_additional_info = checkpoint['my_additional_info']
        # return checkpoint
        pass

    def save_checkpoint(self, filepath):
        self.__save_network(network=self.network, filepath = filepath)

        if self.ema_scheduler is not None:
            self.__save_network(network=self.network_ema, filepath = filepath + '.ema')

        self.__save_training_state(filepath)

    def __save_network(self, network, filepath):
        save_path = filepath

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def __save_training_state(self, filepath):

        assert isinstance(self.optimizers, list) and isinstance(
            self.schedulers, list), 'optimizers and schedulers must be a list.'

        state = {
            'epoch': self.epoch,
            'step': self.step,
            'schedulers': [],
            'optimizers': []
        }

        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())

        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        save_filename = filepath + f'-{self.current_epoch}.state'

        torch.save(state, save_filename)

    def load_checkpoint(self, filepath):
        self.__load_network(
            network=self.network,
            strict=False,
            filepath=filepath
        )

        if self.ema_scheduler is None:
            return

        self.__load_network(
            network=self.network_ema,
            strict=False,
            filepath=filepath + '.ema'
        )

    def __load_network(self, network, filepath, strict=True):
        if not os.path.exists(filepath):
            return

        load_path = filepath

        network.load_state_dict(
            torch.load(load_path),
            strict=strict
        )


class DiffusarData(pl.LightningDataModule):
    def __init__(self, train_images_path, train_annotations_path, batch_size, dataloader_workers, val_images_path = None, val_annotations_path = None):
        super().__init__()

        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        self.train_images_path = train_images_path
        self.train_annotations_path = train_annotations_path

        self.val_images_path = None
        self.val_annotations_path = None

        if val_images_path is not None or val_annotations_path is not None:
            self.val_images_path = val_images_path
            self.val_annotations_path = val_annotations_path

    def setup(self):
        self.train_dataset = ArtifactDataset(
            images_path=self.train_images_path,
            annotations_path=self.train_annotations_path,
        )
        
        if self.val_images_path is None or self.val_annotations_path is None:
            self.val_dataset = None
            return 
        
        self.val_dataset = ArtifactDataset(
            images_path=self.val_images_path,
            annotations_path=self.val_annotations_path,
        )

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.dataloader_workers)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.dataloader_workers)