from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
import lightning_model as lm
import lightning_data as ld
import lightning_progress_bar as lpb
import wandb
import yaml


# Configs
config = yaml.safe_load(open('./config.yaml'))

model_config = config['training']['model_config_path']
model_config = yaml.safe_load(open(model_config))

secrets = config['path']['secrets_path']
secrets = yaml.safe_load(open(secrets))

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
callbacks = []
ckpt_callback_args: dict = config['callbacks']['model_checkpoint']

if config['callbacks']['telegram_tqdm']['use']:
    chat_id = secrets['telegram_tqdm']['chat_id']
    token = secrets['telegram_tqdm']['tg_token']
    progress_bar = lpb.TelegramProgressBar(chat_id, token)
    callbacks.append(progress_bar)


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
    wandb.login(key = secrets['wandb']['wandb_key'])
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
    callbacks.append(checkpoint_callback)

    #   Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_args
    )

    logger.experiment.config['model_config'] = model_config
    logger.experiment.config['config'] = config

    model.train()
    trainer.fit(model=model, datamodule=datamodule)
