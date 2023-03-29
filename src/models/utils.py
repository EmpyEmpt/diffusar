# I want to kill myself
import sys
sys.path.insert(0, './src/data')
from make_dataset import ArtifactDataset

from palette.loss import mse_loss
from torch.utils.data import DataLoader
from palette.network import Network
import numpy as np
import torch
import random


def get_unet(name, unet_args):
    if name == 'sr3':
        from palette.guided_diffusion_modules.unet import UNet as UNetSR3
        return UNetSR3(**unet_args)
    if name == 'gd':
        from palette.guided_diffusion_modules.unet import UNet as UNetGD
        return UNetGD(**unet_args)
    raise NotImplementedError(
        f'We only have "sr3" and "gd" UNets.... not {name}')


def get_loss_fn():
    # TODO: add support for other losses
    return mse_loss


def get_network(unet_name, unet_args, scheduler_args):
    unet = get_unet(unet_name, unet_args)
    network = Network(unet, scheduler_args)
    network.init_weights()
    return network


def get_dataloader(images_path, annotations_path, dataloader_workers, batch_size):
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


def get_optimizer(network, name, opt_args):
    if name == 'adam':
        from torch.optim import Adam
        opt = Adam(list(filter(lambda p: p.requires_grad,
                   network.parameters())), **opt_args)
        return opt
    if name == 'adamw':
        from torch.optim import AdamW
        opt = AdamW(list(filter(lambda p: p.requires_grad,
                    network.parameters())), **opt_args)
        return opt
    raise NotImplementedError(
        f'We only have "adam" and "adamw" optimizers.... not {name}')


def set_seed(seed, gl_seed=0):
	if seed >= 0 and gl_seed >= 0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	# ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
	# 	speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	# if seed >= 0 and gl_seed >= 0:  # slower, more reproducible
	# 	torch.backends.cudnn.deterministic = True
	# 	torch.backends.cudnn.benchmark = False
	# else:  # faster, less reproducible
	# 	torch.backends.cudnn.deterministic = False
	# 	torch.backends.cudnn.benchmark = True
