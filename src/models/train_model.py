from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.make_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger

import yaml

# Configs
config = yaml.safe_load('../../config.yaml')

resume_path = config['models']['control_path']
batch_size = config['training']['batch_size']
logger_freq = config['training']['logger_freq']
learning_rate = config['training']['learning_rate']
sd_locked = config['training']['sd_locked']
only_mid_control = config['training']['only_mid_control']
precision = config['training']['precision']
accumulate_grad_batches = config['training']['accumulate_grad_batches']


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config['models']['cldm_path']).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

if config['wandb']['use']:
    logger = WandbLogger(
        log_model=config['wandb']['log_model'], project=config['wandb']['project_name'])
    logger.watch(model, log=config['wandb']['log'], log_freq=logger_freq)

else:
    logger = ImageLogger(batch_frequency=logger_freq)


# Dataset and trainer init
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0,
                        batch_size=batch_size, shuffle=True)

trainer = pl.Trainer(logger=logger,
                     gpus=1,
                     precision=precision,
                     accumulate_grad_batches=accumulate_grad_batches)


# Train!
trainer.fit(model, dataloader)
