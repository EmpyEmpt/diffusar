# I want to kill myself
import sys
sys.path.insert(0, './src/data')

from make_dataset import ArtifactDataset

import yaml
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import core.util as Util
from palette.model import Palette
from torch.optim import Adam
from palette.loss import mse_loss
from palette.network import Network
from palette.guided_diffusion_modules.unet import UNet
from palette.network import make_beta_schedule


# Configs
config = yaml.safe_load(open('./config.yaml'))

#   Model init
resume_path: str                = config['models']['control_path']
if os.path.isfile(config['models']['last_path']):
    resume_path: str            = config['models']['last_path']
final_path: str                 = config['models']['final_path']

# Training
# :p

#   Dataloader
batch_size: int                 = config['training']['dataloader']['batch_size']
learning_rate: float            = float(config['training']['dataloader']['learning_rate'])

#   Trainer config
# trainer_args: dict              = config['training']['trainer']

#   Callbacks
# ckpt_callback_args: dict        = config['callbacks']['model_checkpoint']

# Dataset
images_path: str                = config['dataset']['images_dir']
annotations_path: str           = config['dataset']['annotations_path']
dataloader_workers: int         = config['training']['dataloader']['dataloader_workers']

# Logger callback
# logger = WandbLogger(
#     log_model=config['wandb']['log_model'],
#     project=config['wandb']['project_name'],
#     name=config['wandb']['name']
# )

# logger.watch(
#     model, 
#     log=config['wandb']['log'],
#     log_freq=config['wandb']['log_freq']
# )


# Callbacks
# checkpoint_callback = ModelCheckpoint(
#     **ckpt_callback_args
# )

# trainer = pl.Trainer(
#     logger=logger,
#     callbacks=checkpoint_callback,
#     **trainer_args
# )

# Train!
# trainer.fit(model, dataloader)

# Copied from 'colorization_mirflickr25k.json'
model_config = yaml.safe_load(open('./model_config.yaml'))
unet_args = model_config['unet_args']
ema_scheduler = model_config['ema_scheduler_args']
optimizer_args = model_config['optimizer_args']
scheduler_args = model_config['scheduler_args']
training_args = model_config['training']
paths = model_config['paths']
seed = model_config['seed']

def get_Unet():
    # For now it's just guided_diffusion
    # TODO: add support for sr3
    Unet = UNet(**unet_args)
    return Unet

def get_beta_scheduler():
    scheduler = make_beta_schedule(**scheduler_args)
    return scheduler

def get_loss_fn():
    # For now it's just mse_loss
    # TODO: add support for other losses
    return mse_loss

def get_network():
    unet = get_Unet()
    beta_scheduler = get_beta_scheduler()
    network = Network(unet, beta_scheduler)
    network.init_weights()
    return network

def get_dataloader():
    dataset = ArtifactDataset(
        images_path=images_path,
        annotations_path=annotations_path,
    )

    dataloader = DataLoader(
        dataset,
        num_workers=dataloader_workers,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


def get_optimizer(network, opt_args):
    opt = Adam(list(filter(lambda p: p.requires_grad, network.parameters())), **opt_args)
    return opt

def main():

    # set seed and and cuDNN environment
    # torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(seed)

    optimizer = get_optimizer(network, optimizer_args)
    network = get_network()
    loss_fn = get_loss_fn()
    dataloader = get_dataloader()

    model = Palette(
        network = network, 
        loss = loss_fn,
        optimizer=optimizer,
        main_loader=dataloader,
        val_loader=None,
        ema_scheduler=ema_scheduler,
        **training_args,
        **paths
    )

    model.train()


if __name__ == '__main__':
    main()
