import torch
import utils
import os
import copy
import os
from pytorch_lightning import LightningModule


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Diffusar(LightningModule):
    def __init__(self, unet_name, unet_args, scheduler_args, ema_scheduler, optimizer_name, optimizer_args, loss_name, batch_size=1):
        super().__init__()

        self.unet_name = unet_name
        self.unet_args = unet_args
        self.scheduler_args = scheduler_args
        self.ema_scheduler = ema_scheduler
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.loss_name = loss_name
        self.batch_size = batch_size

        self.network = utils.get_network(
            self.unet_name, self.unet_args, self.scheduler_args)
        self.network.init_weights()
        self.loss_fn = utils.get_loss_fn(loss_name)

        self.network.loss_fn = self.loss_fn
        self.network.set_new_noise_schedule(device=self.device)

        self.network.train()

        self.step = 0
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.network_EMA = copy.deepcopy(self.network)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
            self.network_EMA = self.network_EMA.to(self.device)

    def configure_optimizers(self):
        optimizer = utils.get_optimizer(
            self.network, self.optimizer_name, self.optimizer_args)
        return optimizer

    # def forward(self, x):
    #     # x = torch.relu(self.fc1(x))
    #     # x = self.fc2(x)
    #     # return x
    #     pass

    def training_step(self, batch, batch_idx):
        self.step += 1

        self.source_image = batch['source_image']
        self.target_image = batch['target_image']

        loss = self.network(self.target_image, self.source_image)

        if self.step > self.ema_scheduler['ema_start'] and self.step % self.ema_scheduler['ema_step'] == 0:
            self.EMA.update_model_average(self.network_EMA, self.network)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = nn.functional.mse_loss(y_hat, y)
        # self.log('val_loss', loss)
        pass

    def predict_step(self, batch, batch_idx):
        out = []
        for im in batch:
            y_t, ret_arr = self.network.restoration(im)
            out.append([y_t, ret_arr])
        return out

    def save_checkpoint(self, filepath):
        self.__save_network(network=self.network, filepath=filepath)

        if self.ema_scheduler is not None:
            self.__save_network(network=self.network_EMA,
                                filepath=filepath + '.ema')

    def __save_network(self, network, filepath):
        save_path = filepath

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def load_checkpoint(self, filepath):
        self.__load_network(
            network=self.network,
            strict=False,
            filepath=filepath
        )

        if self.ema_scheduler is None:
            return

        self.__load_network(
            network=self.network_EMA,
            strict=False,
            filepath=filepath + '.ema'
        )

    def __load_network(self, network, filepath, strict=True):
        if not os.path.exists(filepath):
            return

        load_path = filepath

        network.load_state_dict(
            torch.load(load_path),
            strict=strict
        )
