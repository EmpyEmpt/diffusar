import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from palette.model import Palette
import utils

# Configs
config = yaml.safe_load(open('./config.yaml'))
model_config = config['training']['model_config_path']
model_config = yaml.safe_load(open(model_config))

#   Model args
paths: dict[str, str] = config['models']

#   Unet
unet_name: str = model_config['unet']['name']
unet_args: dict = model_config['unet']['unet_args']

#   EMA
ema_scheduler: dict = model_config['ema_scheduler']['ema_scheduler_args']
if not model_config['ema_scheduler']['use']:
    ema_scheduler = None

optimizer_name: str = model_config['optimizer']['name']
optimizer_args: dict = model_config['optimizer']['optimizer_args']
scheduler_args: dict = model_config['scheduler_args']

# Misc
seed: int = config['misc']['seed']

# Training
training_args: dict = config['training']['training_args']

#   Dataloader
batch_size: int = config['training']['dataloader']['batch_size']
dataloader_workers: int = config['training']['dataloader']['dataloader_workers']


#   Trainer config
# trainer_args: dict              = config['training']['trainer']

#   Callbacks
# ckpt_callback_args: dict        = config['callbacks']['model_checkpoint']

# Dataset
images_path: str = config['dataset']['images_dir']
annotations_path: str = config['dataset']['annotations_path']

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


def main():
    # torch.backends.cudnn.enabled = True
    utils.set_seed(seed)

    network = utils.get_network(unet_name, unet_args, scheduler_args)
    optimizer = utils.get_optimizer(network, optimizer_name, optimizer_args)
    loss_fn = utils.get_loss_fn()
    dataloader = utils.get_dataloader(
        images_path, annotations_path, dataloader_workers, batch_size)

    model = Palette(
        network=network,
        loss=loss_fn,
        optimizer=optimizer,
        main_loader=dataloader,
        val_loader=None,
        ema_scheduler=ema_scheduler,
        phase='train',
        **training_args,
        **paths
    )

    model.train()
    # model.save_everything()
    # print(model.netG_EMA)


if __name__ == '__main__':
    main()
