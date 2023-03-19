# I want to kill myself
import sys
sys.path.insert(0, './src/data')

from make_dataset import ArtifactDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import save
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import yaml

# Configs
config = yaml.safe_load(open('./config.yaml'))

# Model
resume_path: str                = './models/ckeckpoints/epoch=0-step=5999.ckpt'
final_path: str                 = config['models']['final_path']

# Training
batch_size: int                 = config['training']['batch_size']
learning_rate: float            = float(config['training']['learning_rate'])
sd_locked: bool                 = config['training']['sd_locked']
only_mid_control: bool          = config['training']['only_mid_control']
precision: int                  = config['training']['precision']
accumulate_grad_batches: int    = config['training']['accumulate_grad_batches']

# Callbacks
ckpt_freq: int                  = config['callbacks']['model_checkpoint']['ckpt_freq']
ckpt_dir: str                   = config['callbacks']['model_checkpoint']['ckpt_dir']
logger_freq: int                = config['callbacks']['model_checkpoint']['logger_freq']
save_last: bool                 = config['callbacks']['model_checkpoint']['save_last']
save_weight_only: bool          = config['callbacks']['model_checkpoint']['save_weight_only']
save_top_k: int                 = config['callbacks']['model_checkpoint']['save_top_k']
monitor_mc: str                 = config['callbacks']['model_checkpoint']['monitor']

monitor_es: str                 = config['callbacks']['early_stopping']['monitor']
patience: int                   = config['callbacks']['early_stopping']['patience']

# Dataset
images_path: str                = config['dataset']['images_dir']
annotations_path: str           = config['dataset']['annotations_path']
dataloader_workers: int         = config['training']['dataloader_workers']

# First use cpu to load models. 
model = create_model(config['models']['cldm_path']).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

if config['wandb']['use']:
    logger = WandbLogger(
        log_model=config['wandb']['log_model'], 
        project=config['wandb']['project_name'], 
        name=config['wandb']['name'])
    
    logger.watch(model, log=config['wandb']['log'], log_freq=logger_freq)

else:
    logger = ImageLogger(batch_frequency=logger_freq)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor=monitor_mc,
    verbose=False,
    save_last=save_last,
    save_top_k=save_top_k,
    save_weights_only=save_weight_only,
    every_n_train_steps=ckpt_freq)

early_stop_callback = EarlyStopping(
    monitor=monitor_es,
    patience=patience,
    mode='min'
)

# Dataset and trainer init
dataset = ArtifactDataset(images_path = images_path, annotations_path = annotations_path, use_prompts = True)
dataloader = DataLoader(
    dataset, 
    num_workers=dataloader_workers,
    batch_size=(batch_size // 2) * 50, 
    shuffle=True)


trainer = pl.Trainer(
    logger=logger,
    accelerator='gpu',
    gpus = 1,
    precision=precision,
    max_epochs = 1000,
    accumulate_grad_batches = 50,
    callbacks=[checkpoint_callback, early_stop_callback])


# Train!
trainer.fit(model, dataloader)

save(model.state_dict(), final_path)