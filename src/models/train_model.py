from share import *

# I want to kill myself
import sys
sys.path.insert(0, './src/data')

from make_dataset import MyDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger

import yaml

# Configs
config = yaml.safe_load(open('./config.yaml'))

# Training
resume_path: str                = config['models']['control_path']
batch_size: int                 = config['training']['batch_size']
logger_freq: int                = config['training']['logger_freq']
learning_rate: float            = float(config['training']['learning_rate'])
sd_locked: bool                 = config['training']['sd_locked']
only_mid_control: bool          = config['training']['only_mid_control']
precision: int                  = config['training']['precision']
accumulate_grad_batches: int    = config['training']['accumulate_grad_batches']

# Dataset
images_path: str                = config['dataset']['images_dir']
annotations_path: str           = config['dataset']['annotations_path']
dataloader_workers: int         = config['training']['dataloader_workers']

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config['models']['cldm_path']).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

if config['wandb']['use']:
    logger = WandbLogger(
        log_model=config['wandb']['log_model'], project=config['wandb']['project_name'], name=config['wandb']['name'])
    logger.watch(model, log=config['wandb']['log'], log_freq=logger_freq)

else:
    logger = ImageLogger(batch_frequency=logger_freq)


# Dataset and trainer init
dataset = MyDataset(images_path, annotations_path)
dataloader = DataLoader(dataset, num_workers=dataloader_workers,
                        batch_size=batch_size, shuffle=True)


trainer = pl.Trainer(logger=logger,
                    #  accelerator='cpu',
                     precision=precision,
                    #  accumulate_grad_batches=accumulate_grad_batches
                     )


# Train!
trainer.fit(model, dataloader)
