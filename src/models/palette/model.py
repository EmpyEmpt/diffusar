# I'm not doing anything that requires a mask so I'm commenting out all the code where 'mask' or 'mask_image' is present
import torch
import tqdm
from core.base_model import BaseModel
import copy


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


class Palette(BaseModel):
    def __init__(self, network, loss, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = loss
        self.netG = network
        self.ema_scheduler = None

        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])

        self.netG = self.set_device(self.netG, distributed=False)
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=False)

        self.phase_loader = None
        self.load_networks()

        self.optG = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])

        self.optimizers.append(self.optG)
        self.resume_training()

        self.netG.set_loss(self.loss_fn)
        self.netG.set_new_noise_schedule(phase=self.phase)

        self.sample_num = sample_num
        self.task = task

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.source_image = self.set_device(data.get('source_image'))
        self.target_image = self.set_device(data.get('target_image'))
        # self.mask = self.set_device(data.get('mask'))
        # self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'target_image': (self.target_image.detach()[:].float().cpu()+1)/2,
            'source_image': (self.source_image.detach()[:].float().cpu()+1)/2,
        }
        # if self.task in ['inpainting', 'uncropping']:
        #     dict.update({
        #         'mask': self.mask.detach()[:].float().cpu(),
        #         'mask_image': (self.mask_image+1)/2,
        #     })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append(f'target_{self.path[idx]}')
            ret_result.append(self.target_image[idx].detach().float().cpu())

            ret_path.append(f'Process_{self.path[idx]}')
            ret_result.append(
                self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append(f'Out_{self.path[idx]}')
            ret_result.append(
                self.visuals[idx-self.batch_size].detach().float().cpu())

        # if self.task in ['inpainting', 'uncropping']:
        #     ret_path.extend(['Mask_{}'.format(name) for name in self.path])
        #     ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(
            name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):

        self.netG.train()

        for train_data in tqdm.tqdm(self.phase_loader):

            self.set_input(train_data)
            self.optG.zero_grad()
            # Since I'm not using any masks - I do not need it
            loss = self.netG(self.target_image, self.source_image)
            # loss = self.netG(self.target_image, self.source_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size

            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()

        return

    def val_step(self):
        self.netG.eval()

        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)

                # if self.task in ['inpainting', 'uncropping']:
                #     self.output, self.visuals = self.netG.restoration(
                #         self.source_image,
                #         y_t=self.source_image,
                #         y_0=self.target_image,
                #         mask=self.mask,
                #         sample_num=self.sample_num
                #     )
                # else:
                #     self.output, self.visuals = self.netG.restoration(
                #         self.source_image,
                #         sample_num=self.sample_num
                #     )
                self.output, self.visuals = self.netG.restoration(
                    self.source_image,
                    sample_num=self.sample_num
                )

                self.iter += self.batch_size

        return

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)

                # if self.task in ['inpainting', 'uncropping']:
                #     self.output, self.visuals = self.netG.restoration(
                #         self.source_image,
                #         y_t=self.source_image,
                #         y_0=self.target_image,
                #         mask=self.mask,
                #         sample_num=self.sample_num
                #     )
                # else:
                #     self.output, self.visuals = self.netG.restoration(
                #         self.source_image,
                #         sample_num=self.sample_num
                #     )
                self.output, self.visuals = self.netG.restoration(
                    self.source_image,
                    sample_num=self.sample_num
                )
                self.iter += self.batch_size

    def load_networks(self):
        netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG,
                          network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA,
                              network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """

        netG_label = self.netG.__class__.__name__

        self.save_network(network=self.netG, network_label=netG_label)

        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA,
                              network_label=netG_label+'_ema')

        self.save_training_state()
