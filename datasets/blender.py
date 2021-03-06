from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

from .ray_utils import *


def add_perturbation(img, perturbation, seed):
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)
    return img


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(400, 400), perturbation=[], is_learning_density=True, render_patches=False):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.is_learning_density = is_learning_density
        self.render_patches = render_patches

        assert set(perturbation).issubset({"color", "occ"}), \
            'Only "color" and "occ" perturbations are supported!'
        self.perturbation = perturbation
        self.read_meta()
        self.white_back = True

    def set_params(self, is_learning_density, render_patches):
        self.is_learning_density = is_learning_density
        self.render_patches = render_patches

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split.split('_')[-1]}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train':  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for t, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                if t != 0: # perturb everything except the first image.
                           # cf. Section D in the supplementary material
                    img = add_perturbation(img, self.perturbation, t)

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            if self.is_learning_density:
                return len(self.all_rays)
            else:
                return len(self.all_rays) // 2500 if self.render_patches else 100
        if self.split == 'val':
            return 8  # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            if self.is_learning_density:
                sample = {'rays': self.all_rays[idx, :8],
                          'ts': self.all_rays[idx, 8].long(),
                          'rgbs': self.all_rgbs[idx]}
            else:
                if self.render_patches:
                    img_wh = self.img_wh[0]
                    sqrt_n_of_patches = (img_wh // 50)
                    n_of_patches = sqrt_n_of_patches ** 2

                    img_idx = idx // n_of_patches
                    rays = self.all_rays[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2), :8]
                    rays = rays.view(img_wh, img_wh, 8)
                    ts = self.all_rays[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2), 8]
                    ts = ts.view(img_wh, img_wh)
                    rgbs = self.all_rgbs[img_idx * (img_wh ** 2):(img_idx + 1) * (img_wh ** 2)]
                    rgbs = rgbs.view(img_wh, img_wh, 3)

                    patch_idx = idx % n_of_patches
                    i, j = patch_idx // sqrt_n_of_patches, patch_idx % sqrt_n_of_patches  # patch row and column
                    sample = {'rays': rays[i * 50:(i + 1) * 50, j * 50:(j + 1) * 50, :].reshape(2500, 8),
                              'ts': ts[i * 50:(i + 1) * 50, j * 50:(j + 1) * 50].flatten().long(),
                              'rgbs': rgbs[i * 50:(i + 1) * 50, j * 50:(j + 1) * 50, :].reshape(2500, 3)}
                else:
                    img_size = self.img_wh[0] ** 2

                    sample = {'rays': self.all_rays[img_size * idx: img_size * (idx + 1), :8],
                              'ts': self.all_rays[img_size * idx: img_size * (idx + 1), 8].long(),
                              'rgbs': self.all_rgbs[img_size * idx: img_size * (idx + 1)]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            t = idx # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.split == 'test_train' and idx != 0:
                t = idx
                img = add_perturbation(img, self.perturbation, idx)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

            if self.split == 'test_train' and self.perturbation:
                 # append the original (unperturbed) image
                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, H, W)
                valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask

        return sample