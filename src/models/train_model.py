# I want to kill myself
import sys
sys.path.insert(0, './src/data')

from make_dataset import ArtifactDataset
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

# Model init
resume_path: str                = config['models']['control_path']
if os.path.isfile(config['models']['last_path']):
    resume_path: str            = config['models']['last_path']
final_path: str                 = config['models']['final_path']
cldm_path: str                  = config['models']['cldm_path']

# Training
#   Model
sd_locked: bool                 = config['training']['model']['sd_locked']
only_mid_control: bool          = config['training']['model']['only_mid_control']

#   Dataloader
batch_size: int                 = config['training']['dataloader']['batch_size']
learning_rate: float            = float(config['training']['dataloader']['learning_rate'])

#   Trainer config
trainer_args: dict              = config['training']['trainer']

# Callbacks
ckpt_callback_args: dict        = config['callbacks']['model_checkpoint']

# Dataset
images_path: str                = config['dataset']['images_dir']
annotations_path: str           = config['dataset']['annotations_path']
dataloader_workers: int         = config['training']['dataloader']['dataloader_workers']

# First use cpu to load models. 
model = create_model(cldm_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Creating logger callback
if config['wandb']['use']:
    logger = WandbLogger(
        log_model=config['wandb']['log_model'], 
        project=config['wandb']['project_name'], 
        name=config['wandb']['name'])
    
    logger.watch(model, log=config['wandb']['log'], log_freq=config['wandb']['log_freq'])
else:
    logger = ImageLogger(batch_frequency=config['wandb']['log_freq'])

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

trainer = pl.Trainer(
    logger=logger,
    callbacks=checkpoint_callback,
    **trainer_args
    )


# Train!
trainer.fit(model, dataloader)