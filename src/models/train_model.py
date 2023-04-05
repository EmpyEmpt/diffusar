from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
import lightning_model as lm
import lightning_data as ld
import yaml


# Configs
config = yaml.safe_load(open('./config.yaml'))
model_config = config['training']['model_config_path']
model_config = yaml.safe_load(open(model_config))

#   Unet
unet_name: str = model_config['unet']['name']
unet_args: dict = model_config['unet']['unet_args']

#   EMA
ema_scheduler: dict = model_config['ema_scheduler']['ema_scheduler_args']
if not model_config['ema_scheduler']['use']:
    ema_scheduler = None

#   Optimizer
optimizer_name: str = model_config['optimizer']['name']
optimizer_args: dict = model_config['optimizer']['optimizer_args']

#   Scheduler
scheduler_args: dict = model_config['scheduler_args']

#   Loss
loss_name: str = model_config['loss']['loss_name']

#   Trainer config
trainer_args: dict = config['training']['trainer']

#   Dataloader
batch_size: int = config['training']['dataloader']['batch_size']
dataloader_workers: int = config['training']['dataloader']['dataloader_workers']

#   Dataset
images_path: str = config['dataset']['images_dir']
annotations_path: str = config['dataset']['annotations_path']

#   Callbacks
ckpt_callback_args: dict = config['callbacks']['model_checkpoint']

#   Misc things
paths: dict = config['paths']
seed: int = config['misc']['seed']

# Train!
if __name__ == '__main__':
    seed_everything(seed)

    #   Model init
    model = lm.Diffusar(
        unet_name=unet_name,
        unet_args=unet_args,
        scheduler_args=scheduler_args,
        ema_scheduler=ema_scheduler,
        optimizer_name=optimizer_name,
        optimizer_args=optimizer_args,
        loss_name=loss_name,
        batch_size=batch_size
    )

    #   Datamodule init
    datamodule = ld.DiffusarData(
        train_images_path=images_path,
        train_annotations_path=annotations_path,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers
    )

    #   Wandb logger callback
    logger = WandbLogger(
        log_model=config['wandb']['log_model'],
        project=config['wandb']['project_name'],
        name=config['wandb']['name']
    )

    logger.watch(
        model=model,
        log=config['wandb']['log'],
        log_freq=config['wandb']['log_freq']
    )

    #   Callbacks
    checkpoint_callback = ModelCheckpoint(
        **ckpt_callback_args
    )

    #   Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=checkpoint_callback,
        **trainer_args
    )

    trainer.logger.experiment.log(config)
    trainer.logger.experiment.log(model_config)

    model.train()
    trainer.fit(model=model, datamodule=datamodule)
