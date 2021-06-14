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


def show(img):
    plt.imshow(img.detach().cpu().numpy())
    plt.show()


class Pipeline:
    def __init__(self, hparams):
        super().__init__()

        # COARSE_PATH = 'ckpts/density_nerf_coarse_epoch-0.pt'
        # FINE_PATH = 'ckpts/density_nerf_fine_epoch-0.pt'

        torch.manual_seed(1337)

        self.hparams = hparams
        self.is_learning_density = True
        self.device = torch.device('cuda') if hparams.use_gpu else torch.device('cpu')

        self.style_image = Image.open('style.jpg')
        self.style_image = torch.tensor(np.array(self.style_image), device=self.device, dtype=torch.float)
        self.style_image = self.style_image.permute(2, 0, 1)
        self.style_image = torch.unsqueeze(self.style_image, 0) / 255.0

        self.embeddings = {
            'xyz': PosEmbedding(hparams.N_emb_xyz - 1, hparams.N_emb_xyz).to(self.device),
            'dir': PosEmbedding(hparams.N_emb_dir - 1, hparams.N_emb_dir).to(self.device)
        }
        self.models = {
            'coarse': NeRF('coarse').to(self.device),
            'fine': NeRF('fine', beta_min=hparams.beta_min).to(self.device)
        }

        # self.models['coarse'].load_state_dict(torch.load(COARSE_PATH))
        # self.models['fine'].load_state_dict(torch.load(FINE_PATH))

        self.dataset = dataset_dict[hparams.dataset_name]
        self.kwargs = {'root_dir': hparams.root_dir,
                       'img_wh': tuple(hparams.img_wh),
                       'perturbation': hparams.data_perturb}
        if hparams.dataset_name == 'llff':
            self.kwargs['spheric_poses'] = hparams.spheric_poses
            self.kwargs['val_num'] = hparams.num_gpus

        self.val_dataset = self.dataset(split='val', **self.kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)

        self.logger = Logger()

    def training_data_setup(self):
        if self.is_learning_density:
            self.train_dataset = self.dataset(split='train', is_learning_density=True, **self.kwargs)
            self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, num_workers=4,
                                               batch_size=self.hparams.batch_size, pin_memory=True)
        else:
            self.train_dataset = self.dataset(split='train', is_learning_density=False, **self.kwargs)
            self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=1,
                                               pin_memory=True)

    def create_checkpoint_data(self):
        dataset = self.dataset(split='train', is_learning_density=True, **self.kwargs)
        img_idx = 4
        img_wh = tuple(self.hparams.img_wh)[0]
        self.checkpoint_rays = dataset.all_rays[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2), :8]
        self.checkpoint_ts = dataset.all_rays[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2), 8]

    def switch_stage(self):
        self.is_learning_density = not self.is_learning_density
        for child in self.models['coarse'].children():
            if hasattr(child, 'density'):
                for param in child.parameters():
                    param.requires_grad = not param.requires_grad
        for child in self.models['fine'].children():
            if hasattr(child, 'density'):
                for param in child.parameters():
                    param.requires_grad = not param.requires_grad

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

    def train(self):
        is_learning_density = self.is_learning_density
        loss_func = loss_dict['nerfw'](coef=1) if is_learning_density else loss_dict['style'](self.style_image)
        num_epochs = self.hparams.num_epochs_density if is_learning_density else self.hparams.num_epochs_style

        device = self.device
        logger = self.logger

        self.hparams.lr = 5e-4 if self.is_learning_density else 1e-3

        optimizer = get_optimizer(self.hparams, self.models)
        # scheduler = get_scheduler(hparams, optimizer)

        self.training_data_setup()

        self.create_checkpoint_data()
        self.models['coarse'].train()
        self.models['fine'].train()

        for epoch in range(num_epochs):
            # training
            print(f'Starting epoch {epoch+1}...')

            for idx, batch in enumerate(self.train_dataloader):
                rays, rgbs, ts = batch['rays'].to(device), batch['rgbs'].to(device), batch['ts'].to(device)
                if torch.mean(rgbs) > 0.99 and not self.is_learning_density:  # empty image
                    continue
                if not self.is_learning_density:
                    rays = rays.squeeze()  # (H*W, 3)
                    rgbs = rgbs.squeeze()  # (H*W, 3)
                    ts = ts.squeeze()

                results = self(rays, ts, self.train_dataloader.dataset.white_back)

                optimizer.zero_grad()
                loss_d = loss_func(results, rgbs)
                for k, v in loss_d.items():
                    logger(f'Loss/train_{k}', v)
                loss = sum(l for l in loss_d.values())
                logger('Loss/train', loss)
                loss.backward()
                optimizer.step()

                if self.is_learning_density:
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    with torch.no_grad():
                        logger('PSNR/train', psnr(results[f'rgb_{typ}'], rgbs))

                if idx % 1000 == 0:
                    self.models['coarse'].eval()
                    self.models['fine'].eval()
                    with torch.no_grad():
                        rays, ts = self.checkpoint_rays.to(device), self.checkpoint_ts.to(device)
                        rays = rays.squeeze()
                        ts = ts.squeeze()
                        results = self(rays, ts, self.val_dataloader.dataset.white_back)
                        log_image = results['rgb_fine']
                        dim = int(math.sqrt(len(log_image)))
                        log_image = log_image.squeeze().permute(1, 0).view(3, dim, dim)
                        logger(f'checkpoint_image_{idx}', log_image)
                    self.models['coarse'].train()
                    self.models['fine'].train()

            logger('lr', get_learning_rate(optimizer))

            # scheduler.step()

            # model saving
            prefix = 'density' if self.is_learning_density else 'style'
            torch.save(self.models['coarse'].state_dict(), f'ckpts/chair_{prefix}_nerf_coarse_epoch-{epoch}.pt')
            torch.save(self.models['fine'].state_dict(), f'ckpts/chair_{prefix}_nerf_fine_epoch-{epoch}.pt')

            # # validation
            # self.models['coarse'].eval()
            # self.models['fine'].eval()
            #
            # list_loss_d, list_loss, list_psnr = [], [], []
            # with torch.no_grad():
            #     for idx, batch in enumerate(self.val_dataloader):
            #         rays, rgbs, ts = batch['rays'].to(device), batch['rgbs'].to(device), batch['ts'].to(device)
            #         rays = rays.squeeze()  # (H*W, 3)
            #         rgbs = rgbs.squeeze()  # (H*W, 3)
            #         ts = ts.squeeze()  # (H*W)
            #         results = self(rays, ts, self.val_dataloader.dataset.white_back)
            #
            #         loss_d = loss_func(results, rgbs)
            #         list_loss_d.append(loss_d)
            #
            #         loss = sum(l for l in loss_d.values())
            #         list_loss.append(loss)
            #
            #         if self.is_learning_density:
            #             list_psnr.append(psnr(results[f'rgb_{typ}'], rgbs))
            #             typ = 'fine' if 'rgb_fine' in results else 'coarse'
                        # if idx == 0:
                        #     W, H = hparams.img_wh
                        #     img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                        #     img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                        #     depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
                        #     stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
                        #     logger('GT_pred_depth/val', stack)

            # for k, v in list_loss_d[0].items():
            #     logger(f'Loss/val_{k}', sum([loss_d[k] for loss_d in list_loss_d]) / len(list_loss_d))
            #
            # logger('Loss/val', sum(list_loss) / len(list_loss))
            # if self.is_learning_density:
            #     logger('PSNR/val', sum(list_psnr) / len(list_psnr))


if __name__ == '__main__':
    hparams = get_opts()
    pl = Pipeline(hparams)
    pl.train()
    pl.switch_stage()
    pl.train()
