import torch
from torch import nn
import math
from torchvision import models
from collections import namedtuple


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class StyleLoss(nn.Module):
    def __init__(self, style_image):
        super().__init__()
        self.style_image = style_image
        device = style_image.device
        self.upsampler = nn.Upsample(size=self.style_image.shape[-2:], mode='bilinear', align_corners=True).to(device)
        self.vgg = Vgg16(requires_grad=False).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.features_style = self.vgg(normalize_batch(self.style_image))
        self.gram_style = [gram_matrix(y) for y in self.features_style]

    def forward(self, inputs, targets):
        rgb_fine = inputs['rgb_fine']
        dim = int(math.sqrt(len(rgb_fine)))
        rgb_fine = rgb_fine.view(1, 3, dim, dim) * 255

        # content loss
        features_rendered = self.vgg(normalize_batch(rgb_fine))
        features_gt = self.vgg(normalize_batch(targets.view(1, 3, dim, dim) * 255))
        content_loss = self.mse_loss(features_rendered.relu2_2, features_gt.relu2_2)

        # style loss
        features_rendered_upsampled = self.vgg(normalize_batch(self.upsampler(rgb_fine)))
        style_loss = 0.
        for ft_y, gm_s in zip(features_rendered_upsampled, self.gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += self.mse_loss(gm_y, gm_s)

        return {
            'ct_l': 1e3 * content_loss,
            'st_l': 1e8 * style_loss
        }


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets):
        ret = {}
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()
            else:
                ret['f_l'] = \
                    ((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean() # +3 to make it positive
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret


loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss,
             'style': StyleLoss}
