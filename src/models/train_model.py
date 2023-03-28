# I want to kill myself
import sys
sys.path.insert(0, './src/data')

from make_dataset import ArtifactDataset
from palette import create_model, define_network, define_loss, define_metric
import core.util as Util
from typing import Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

import yaml

# Configs
config = yaml.safe_load(open('./config.yaml'))

# # Model init
# resume_path: str                = config['models']['control_path']
# if os.path.isfile(config['models']['last_path']):
#     resume_path: str            = config['models']['last_path']
# final_path: str                 = config['models']['final_path']
# cldm_path: str                  = config['models']['cldm_path']

# # Training
# #   Model
# sd_locked: bool                 = config['training']['model']['sd_locked']
# only_mid_control: bool          = config['training']['model']['only_mid_control']

#   Dataloader
batch_size: int                 = config['training']['dataloader']['batch_size']
learning_rate: float            = float(config['training']['dataloader']['learning_rate'])

# #   Trainer config
# trainer_args: dict              = config['training']['trainer']

# # Callbacks
ckpt_callback_args: dict        = config['callbacks']['model_checkpoint']

# Dataset
images_path: str                = config['dataset']['images_dir']
annotations_path: str           = config['dataset']['annotations_path']
dataloader_workers: int         = config['training']['dataloader']['dataloader_workers']

# Creating logger callback

logger = WandbLogger(
    log_model=config['wandb']['log_model'], 
    project=config['wandb']['project_name'], 
    name=config['wandb']['name'])

logger.watch(model, log=config['wandb']['log'], log_freq=config['wandb']['log_freq'])


# Callbacks
checkpoint_callback = ModelCheckpoint(
    **ckpt_callback_args
    )

# Dataloader and trainer
dataset = ArtifactDataset(
    images_path = images_path, 
    annotations_path = annotations_path, 
    use_prompts = True
    )

dataloader = DataLoader(
    dataset, 
    num_workers=dataloader_workers,
    batch_size=batch_size, 
    shuffle=True
    )

# trainer = pl.Trainer(
#     logger=logger,
#     callbacks=checkpoint_callback,
#     **trainer_args
#     )


# # Train!
# trainer.fit(model, dataloader)


# this used to be in config (opt) but I don't want it THAT WAY
phase: str = 'train'
seed: int = 42


def main():

    # set seed and and cuDNN environment
    # torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(seed)

    # TODO: rework network definition
    network_opts: list
    networks = [define_network(network_opt) for network_opt in network_opts]

    ''' set metrics, loss, optimizer and  schedulers '''
    # TODO: rework metrics and loss definition
    metric_opts: list
    loss_opts: list
    metrics = [define_metric(metric_opt) for metric_opt in metric_opts]
    losses = [define_loss(loss_opt) for loss_opt in loss_opts]

    opt: dict
    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=dataloader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
    )

    if phase == 'train':
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    main()
