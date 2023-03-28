# I'm not doing anything that requires a mask so I'm commenting out all the code where 'mask' or 'mask_image' is present
import torch
import tqdm
import copy
import collections
import core.util as Util
import os


CustomResult = collections.namedtuple('CustomResult', 'name result')

# TODO: Add normal paths
checkpoint_path: str
resume_path: str
phase: str
batch_size: int
max_epochs: int
max_steps: int
save_every_n_checkpoints: int
val_every_n_checkpoints: int

# which GPU training is on
global_rank: int


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


class Palette:
    def __init__(self, network, loss, sample_num, task, optimizers, device, main_loader, val_loader, ema_scheduler=None, **kwargs):
        self.loss_fn = loss
        self.netG = network
        self.ema_scheduler = None

        self.main_loader = main_loader
        self.val_loader = val_loader

        self.device = device
        self.schedulers = []
        self.optimizers = []
        self.metrics = []
        self.sample_num = sample_num
        self.task = task
        self.phase = phase
        self.batch_size = batch_size
        self.epoch = 0
        self.step = 0
        self.results_dict = CustomResult([], [])

        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])

        if self.device == 'gpu':
            self.netG = self.netG.cuda()

        if self.ema_scheduler is not None:
            if self.device == 'gpu':
                self.netG_EMA = self.netG_EMA.cuda()

        self.load_networks()

        self.optG = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])

        self.optimizers.append(self.optG)
        self.netG.set_loss(self.loss_fn)
        self.netG.set_new_noise_schedule(phase=self.phase)

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.source_image = data.get('source_image')
        self.target_image = data.get('target_image')

        if self.device == 'gpu':
            self.source_image = self.source_image.cuda()
            self.target_image = self.target_image.cuda()

        # self.mask = self.set_device(data.get('mask'))
        # self.mask_image = data.get('mask_image')
        self.path = data['path']
        # self.batch_size = len(data['path'])

    def train(self):
        while self.epoch <= max_epochs and self.step <= max_steps:
            self.epoch += 1

            self.train_step()

            if self.epoch % save_every_n_checkpoints == 0:
                self.save_everything()

            if self.epoch % val_every_n_checkpoints == 0:
                if self.val_loader is None:
                    pass
                else:
                    self.val_step()

    def train_step(self):

        self.netG.train()

        for train_data in tqdm.tqdm(self.main_loader):

            self.set_input(train_data)
            self.optG.zero_grad()
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

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.main_loader):
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
        self.__load_network(network=self.netG,
                            network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.__load_network(network=self.netG_EMA,
                                network_label=netG_label+'_ema', strict=False)

    def __load_network(self, network, network_label, strict=True):
        if resume_path is None:
            return

        model_path = f"{resume_path}_{network_label}.pth"

        if not os.path.exists(model_path):
            return

        network.load_state_dict(
            torch.load(
                model_path,
                map_location=lambda storage,
                loc: Util.set_device(storage)
            ),
            strict=strict
        )

    def save_everything(self):
        """ load pretrained model and training state. """

        netG_label = self.netG.__class__.__name__

        self.__save_network(network=self.netG, network_label=netG_label)

        if self.ema_scheduler is not None:
            self.__save_network(network=self.netG_EMA,
                                network_label=netG_label+'_ema')

        self.__save_training_state()

    def __save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        if global_rank != 0:
            return

        save_filename = f'{self.epoch}_{network_label}.pth'
        save_path = os.path.join(checkpoint_path, save_filename)

        # if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        #     network = network.module

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def __save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        if global_rank != 0:
            return

        assert isinstance(self.optimizers, list) and isinstance(
            self.schedulers, list), 'optimizers and schedulers must be a list.'

        state = {'epoch': self.epoch, 'step': self.step,
                 'schedulers': [], 'optimizers': []}

        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())

        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        save_filename = f'{self.epoch}.state'
        save_path = os.path.join(checkpoint_path, save_filename)
        torch.save(state, save_path)

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