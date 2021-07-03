from opt import get_opts
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import dataset_dict
from models.nerf import *
from models.rendering import *
from utils import *
from losses import loss_dict
from metrics import *
from utils.logger import Logger
import math
import matplotlib.pyplot as plt
import imageio
import os


def show(img):
    plt.imshow(img.detach().cpu().numpy())
    plt.show()


class Pipeline:
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.logger = Logger()
        torch.manual_seed(1337)

        self.hparams = hparams
        self.device = torch.device('cuda') if hparams.use_gpu else torch.device('cpu')

        self.embeddings = {
            'xyz': PosEmbedding(hparams.N_emb_xyz - 1, hparams.N_emb_xyz).to(self.device),
            'dir': PosEmbedding(hparams.N_emb_dir - 1, hparams.N_emb_dir).to(self.device)
        }
        self.models = {
            'coarse': NeRF('coarse').to(self.device),
            'fine': NeRF('fine', beta_min=hparams.beta_min).to(self.device)
        }

        self.dataset = dataset_dict[hparams.dataset_name]
        self.kwargs = {'root_dir': hparams.root_dir,
                       'img_wh': tuple(hparams.img_wh),
                       'perturbation': hparams.data_perturb}
        if hparams.dataset_name == 'llff':
            self.kwargs['spheric_poses'] = hparams.spheric_poses
            self.kwargs['val_num'] = hparams.num_gpus

        if 'coarse_path' in kwargs and 'fine_path' in kwargs:
            self.models['coarse'].load_state_dict(torch.load(kwargs['coarse_path']))
            self.models['fine'].load_state_dict(torch.load(kwargs['fine_path']))
            self.train_dataset = self.dataset(split='train', is_learning_density=False, render_patches=False,
                                              **self.kwargs)
        else:
            self.train_dataset = self.dataset(split='train', is_learning_density=True, **self.kwargs)

        self.val_dataset = self.dataset(split='val', **self.kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)

        self.create_checkpoint_data()

    def create_checkpoint_data(self):
        dataset = self.dataset(split='train', is_learning_density=True, **self.kwargs)
        img_idx = 4
        img_wh = tuple(self.hparams.img_wh)[0]
        self.checkpoint_rays = dataset.all_rays[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2), :8]
        self.checkpoint_ts = dataset.all_rays[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2), 8]

    def set_requires_grad(self, requires_grad=True):
        for child in self.models['coarse'].children():
            if hasattr(child, 'density'):
                for param in child.parameters():
                    param.requires_grad = requires_grad
        for child in self.models['fine'].children():
            if hasattr(child, 'density'):
                for param in child.parameters():
                    param.requires_grad = requires_grad

    def __call__(self, rays, ts, white_back):
        results = defaultdict(list)
        for i in range(0, rays.shape[0], self.hparams.chunk):
            rendered_ray_chunks = render_rays(self.models,
                                              self.embeddings,
                                              rays[i:i + self.hparams.chunk],
                                              ts[i:i + self.hparams.chunk],
                                              self.hparams.N_samples,
                                              self.hparams.use_disp,
                                              self.hparams.perturb,
                                              self.hparams.noise_std,
                                              self.hparams.N_importance,
                                              self.hparams.chunk,
                                              white_back)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def log_checkpoint_image(self, string):
        with torch.no_grad():
            rays, ts = self.checkpoint_rays.to(self.device), self.checkpoint_ts.to(self.device)
            rays = rays.squeeze()
            ts = ts.squeeze()
            results = self(rays, ts, self.val_dataloader.dataset.white_back)
            log_image = results['rgb_fine']
            dim = int(math.sqrt(len(log_image)))
            log_image = log_image.squeeze().permute(1, 0).view(3, dim, dim)
            self.logger(f'checkpoint_image_{string}', log_image)

    def learn_density(self, **kwargs):
        loss_func = loss_dict['nerfw'](coef=1)
        num_epochs = self.hparams.num_epochs_density
        device = self.device

        optimizer = get_optimizer(self.hparams, self.models)

        self.models['coarse'].train()
        self.models['fine'].train()

        self.train_dataset.set_params(is_learning_density=True, render_patches=False)
        data_loader = DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.hparams.batch_size,
                                 pin_memory=True)

        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1} to learn density...')
            for idx, batch in enumerate(data_loader):
                if idx % 1000 == 0:
                    self.log_checkpoint_image(str(idx))

                rays, rgbs, ts = batch['rays'].to(device), batch['rgbs'].to(device), batch['ts'].to(device)

                optimizer.zero_grad()
                results = self(rays, ts, self.train_dataset.white_back)
                loss_d = loss_func(results, rgbs)
                for k, v in loss_d.items():
                    self.logger(f'Loss/train_{k}', v)
                loss = sum(l for l in loss_d.values())
                self.logger('Loss/train', loss)
                loss.backward()
                optimizer.step()

                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                with torch.no_grad():
                    self.logger('PSNR/train', psnr(results[f'rgb_{typ}'], rgbs))

    def learn_style(self, style_path, style_mode='memory_saving', **kwargs):
        self.style_image = Image.open(style_path)
        self.style_image = torch.tensor(np.array(self.style_image), device=self.device, dtype=torch.float)
        self.style_image = self.style_image.permute(2, 0, 1)
        self.style_image = torch.unsqueeze(self.style_image, 0) / 255.0

        img_wh = self.kwargs['img_wh'][0]

        num_epochs = self.hparams.num_epochs_density
        device = self.device
        optimizer = get_optimizer(self.hparams, self.models)
        loss_func = loss_dict['style'](self.style_image)

        num_patches = (img_wh // 50) ** 2

        if style_mode == 'patch':  # transfer style using patches
            self.train_dataset.set_params(is_learning_density=False, render_patches=True)
            data_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)

            for epoch in range(num_epochs):
                for idx, data in enumerate(data_loader):

                    # image logging
                    if idx % 64 == 0:
                        self.log_checkpoint_image(str(idx))

                    rays, rgbs, ts = data['rays'].to(device), data['rgbs'].to(device), data['ts'].to(device)
                    rays = rays.squeeze()
                    rgbs = rgbs.squeeze()
                    ts = ts.squeeze()
                    optimizer.zero_grad()
                    rendered_image = self(rays, ts, self.train_dataset.white_back)['rgb_fine']
                    if torch.mean(rendered_image) > 0.99:  # empty image
                        continue
                    loss = loss_func(rendered_image, rgbs)
                    self.logger('Loss/train', loss)
                    loss.backward()
                    optimizer.step()

        elif style_mode == 'memory_saving':  # render whole images to transfer the style
            for epoch in range(num_epochs):
                for i in range(100):

                    # image logging
                    self.log_checkpoint_image(str(i))

                    # Substage 1: store gradients
                    self.train_dataset.set_params(is_learning_density=False, render_patches=False)
                    with torch.no_grad():
                        sample = self.train_dataset[i]
                        rays, rgbs, ts = sample['rays'].to(device), sample['rgbs'].to(device), sample['ts'].to(device)
                        rays = rays.squeeze()
                        rgbs = rgbs.squeeze()
                        ts = ts.squeeze()
                        rendered_image = self(rays, ts, self.train_dataset.white_back)['rgb_fine']
                    rendered_image.requires_grad_()
                    loss = loss_func(rendered_image, rgbs)
                    self.logger('Loss/train', loss)
                    loss.backward()
                    gradient = rendered_image.grad.reshape(img_wh, img_wh, 3).clone().detach()

                    # Substage 2: backprop gradients through NeRF
                    self.train_dataset.set_params(is_learning_density=False, render_patches=True)
                    optimizer.zero_grad()
                    for j in range(num_patches):  # iterate over patches
                        sample = self.train_dataset[i*num_patches + j]
                        rays, rgbs, ts = sample['rays'].to(device), sample['rgbs'].to(device), sample['ts'].to(device)
                        rays = rays.squeeze()
                        ts = ts.squeeze()

                        rendered_image = self(rays, ts, self.train_dataset.white_back)['rgb_fine'].reshape(50, 50, 3)
                        if torch.mean(rendered_image) > 0.99:  # empty image
                            continue
                        r, c = j // (img_wh // 50), j % (img_wh // 50)
                        rendered_image.backward(gradient[r * 50: (r + 1) * 50, c * 50: (c + 1) * 50])
                    optimizer.step()

        elif style_mode == 'small':
            self.train_dataset.set_params(is_learning_density=False, render_patches=False)
            data_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=1)
            for epoch in range(num_epochs):
                for idx, data in enumerate(data_loader):

                    # image logging
                    self.log_checkpoint_image(str(idx))

                    rays, rgbs, ts = data['rays'].to(device), data['rgbs'].to(device), data['ts'].to(device)
                    rays = rays.squeeze()
                    rgbs = rgbs.squeeze()
                    ts = ts.squeeze()

                    optimizer.zero_grad()
                    rendered_image = self(rays, ts, self.train_dataset.white_back)['rgb_fine']
                    loss = loss_func(rendered_image, rgbs)
                    self.logger('Loss/train', loss)
                    loss.backward()
                    optimizer.step()

        else:
            raise ValueError('Please enter a valid mode for style transfer.')

    def log_model(self, prefix, suffix):
        try:
            os.mkdir('./ckpts')
        except FileExistsError:
            pass
        torch.save(self.models['coarse'].state_dict(), f'ckpts/{prefix}_nerf_coarse_{suffix}.pt')
        torch.save(self.models['fine'].state_dict(), f'ckpts/{prefix}_nerf_fine_{suffix}.pt')


if __name__ == '__main__':
    hparams = get_opts()  # parse args

    if hparams.stage == 'geometry':  # stage 1
        pl = Pipeline(hparams)
        pl.learn_density()  # learn the scene geometry
        pl.log_model(prefix=hparams.prefix, suffix=hparams.suffix)  # log the model

    elif hparams.stage == 'style':  # stage 2
        pl = Pipeline(hparams,
                      coarse_path=hparams.coarse_path,  # coarse NeRF
                      fine_path=hparams.fine_path  # fine NeRF
                      )
        pl.set_requires_grad(requires_grad=False)  # freeze density layer
        pl.learn_style(style_path=hparams.style_dir, style_mode=hparams.style_mode)  # transfer the style
        pl.log_model(prefix=hparams.prefix, suffix=hparams.suffix)  # log the model

    else:
        raise ValueError('Please enter a valid stage.')
