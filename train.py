from opt import get_opts
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import dataset_dict
from models.nerf import *
from models.rendering import *
from utils import *
from losses import loss_dict
from metrics import *
from torch.utils.tensorboard import SummaryWriter


class Pipeline(nn.Module):
    def __init__(self, models, embeddings, hparams):
        super().__init__()
        self.device = torch.device('cuda') if hparams.use_gpu else torch.device('cpu')
        self.models = models
        self.models['coarse'].to(self.device)
        self.models['fine'].to(self.device)
        self.embeddings = embeddings
        self.embeddings['xyz'].to(self.device)
        self.embeddings['dir'].to(self.device)
        self.hparams = hparams

    def forward(self, rays, ts, white_back):
        results = defaultdict(list)
        for i in range(0, rays.shape[0], self.hparams.chunk):
            rendered_ray_chunks = render_rays(self.models, self.embeddings, rays[i:i + self.hparams.chunk],
                                              ts[i:i + self.hparams.chunk], self.hparams.N_samples,
                                              self.hparams.use_disp, self.hparams.perturb, self.hparams.noise_std,
                                              self.hparams.N_importance, self.hparams.chunk, white_back)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results


def data_setup(hparams):
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
              'img_wh': tuple(hparams.img_wh),
              'perturbation': hparams.data_perturb}
    if hparams.dataset_name == 'llff':
        kwargs['spheric_poses'] = hparams.spheric_poses
        kwargs['val_num'] = hparams.num_gpus

    train_dataset = dataset(split='train', **kwargs)
    val_dataset = dataset(split='val', **kwargs)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=hparams.batch_size,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
    return train_dataloader, val_dataloader


def main(hparams):
    train_step, val_step = 0, 0
    train_dataloader, val_dataloader = data_setup(hparams)  # data loaders

    # pipeline setup
    embeddings = {
        'xyz': PosEmbedding(hparams.N_emb_xyz - 1, hparams.N_emb_xyz),
        'dir': PosEmbedding(hparams.N_emb_dir - 1, hparams.N_emb_dir)
    }
    models = {
        'coarse': NeRF('coarse'),
        'fine': NeRF('fine', in_channels_a=hparams.N_a, in_channels_t=hparams.N_tau, beta_min=hparams.beta_min)
    }
    pl = Pipeline(models, embeddings, hparams)

    loss_func = loss_dict['nerfw'](coef=1)  # loss function
    device = torch.device('cuda') if hparams.use_gpu else torch.device('cpu')  # GPU or CPU

    optimizer = get_optimizer(hparams, pl.models)
    scheduler = get_scheduler(hparams, optimizer)
    writer = SummaryWriter(log_dir=f'runs/{hparams.exp_name}')

    for epoch in range(hparams.num_epochs):
        # training
        pl.models['coarse'].train()
        pl.models['fine'].train()
        for batch in train_dataloader:
            optimizer.zero_grad()

            rays, rgbs, ts = batch['rays'].to(device), batch['rgbs'].to(device), batch['ts'].to(device)
            results = pl(rays, ts, train_dataloader.dataset.white_back)

            loss_d = loss_func(results, rgbs)
            for k, v in loss_d.items():
                writer.add_scalar(f'Loss/train_{k}', v, train_step)

            loss = sum(l for l in loss_d.values())
            writer.add_scalar('Loss/train', float(loss.item()), train_step)
            loss.backward()
            optimizer.step()

            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            with torch.no_grad():
                writer.add_scalar('PSNR/train', psnr(results[f'rgb_{typ}'], rgbs), train_step)
            train_step += 1

        writer.add_scalar('lr', get_learning_rate(optimizer), train_step, epoch+1)

        scheduler.step()

        # validation
        pl.models['coarse'].eval()
        pl.models['fine'].eval()

        list_loss_d, list_loss, list_psnr = [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                rays, rgbs, ts = batch['rays'].to(device), batch['rgbs'].to(device), batch['ts'].to(device)
                rays = rays.squeeze()  # (H*W, 3)
                rgbs = rgbs.squeeze()  # (H*W, 3)
                ts = ts.squeeze()  # (H*W)
                results = pl(rays, ts, val_dataloader.dataset.white_back)

                loss_d = loss_func(results, rgbs)
                list_loss_d.append(loss_d)

                loss = sum(l for l in loss_d.values())
                list_loss.append(loss)

                list_psnr.append(psnr(results[f'rgb_{typ}'], rgbs))

                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                if idx == 0:
                    W, H = hparams.img_wh
                    img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                    depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
                    stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
                    writer.add_images('GT_pred_depth/val', stack, val_step)

        for k, v in list_loss_d[0].items():
            writer.add_scalar(f'Loss/val_{k}', sum([loss_d[k] for loss_d in list_loss_d]) / len(list_loss_d), val_step)

        writer.add_scalar('Loss/val', sum(list_loss) / len(list_loss), val_step)
        writer.add_scalar('PSNR/val', sum(list_psnr) / len(list_psnr), val_step)
        val_step += 1


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
