# I want to kill myself
import sys
sys.path.insert(0, './src/data')
from make_dataset import ArtifactDataset

import yaml
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

# Configs
config = yaml.safe_load(open('./config.yaml'))
model_config = config['training']['model_config_path']
model_config = yaml.safe_load(open(model_config))

#   Model init
paths: dict[str, str]           = config['models']

unet_args: dict                 = model_config['unet_args']
ema_scheduler: dict             = model_config['ema_scheduler_args']
optimizer_args: dict            = model_config['optimizer_args']
scheduler_args: dict            = model_config['scheduler_args']

# Misc
seed: int                       = config['misc']['seed']

# Training
training_args: dict             = config['training']['training_args']

#   Dataloader
batch_size: int                 = config['training']['dataloader']['batch_size']
dataloader_workers: int         = config['training']['dataloader']['dataloader_workers']


#   Trainer config
# trainer_args: dict              = config['training']['trainer']

#   Callbacks
# ckpt_callback_args: dict        = config['callbacks']['model_checkpoint']

# Dataset
images_path: str                = config['dataset']['images_dir']
annotations_path: str           = config['dataset']['annotations_path']

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

def get_unet():
    # For now it's just guided_diffusion
    # TODO: add support for sr3
    Unet = UNet(**unet_args)
    return Unet

def get_loss_fn():
    # For now it's just mse_loss
    # TODO: add support for other losses
    return mse_loss

def get_network():
    unet = get_unet()
    network = Network(unet, scheduler_args)
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

    network = get_network()
    optimizer = get_optimizer(network, optimizer_args)
    loss_fn = get_loss_fn()
    dataloader = get_dataloader()

    model = Palette(
        network = network, 
        loss = loss_fn,
        optimizer=optimizer,
        main_loader=dataloader,
        val_loader=None,
        ema_scheduler=ema_scheduler,
        phase = 'train',
        **training_args,
        **paths
    )

    # model.train()
    model.save_everything()
    # print(model.netG_EMA)


if __name__ == '__main__':
    main()
