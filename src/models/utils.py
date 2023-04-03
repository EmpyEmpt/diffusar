# I want to kill myself
import sys
sys.path.insert(0, './src/data')
from make_dataset import ArtifactDataset

from palette.network import Network

def get_unet(unet_name, unet_args):
    if unet_name == 'sr3':
        from palette.guided_diffusion_modules.unet import UNet as UNetSR3
        return UNetSR3(**unet_args)
    
    if unet_name == 'gd':
        from palette.guided_diffusion_modules.unet import UNet as UNetGD
        return UNetGD(**unet_args)
    
    raise NotImplementedError(
        f'We only have "sr3" and "gd" UNets.... not {unet_name}'
    )


def get_loss_fn(loss_name):
    if loss_name == 'mse':
        from palette.loss import mse_loss
        return mse_loss
    
    if loss_name == 'focal_loss':
        from palette.loss import FocalLoss
        return FocalLoss
    
    raise NotImplementedError(
        f'We only have "mse" and "focal_loss" losses.... not {loss_name}'
    )


def get_network(unet_name, unet_args, scheduler_args):
    unet = get_unet(unet_name, unet_args)
    network = Network(unet, scheduler_args)
    return network


def get_optimizer(network, name, opt_args):
    if name == 'adam':
        from torch.optim import Adam
        opt = Adam(
            list(filter(lambda p: p.requires_grad, network.parameters())),
            **opt_args
        )
        return opt
    if name == 'adamw':
        from torch.optim import AdamW
        opt = AdamW(
            list(filter(lambda p: p.requires_grad, network.parameters())),
            **opt_args
        )
        return opt
    raise NotImplementedError(
        f'We only have "adam" and "adamw" optimizers.... not {name}')
