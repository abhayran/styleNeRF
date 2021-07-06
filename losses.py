import torch
from torch import nn
import math
from torchvision import models
import copy
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss


class FeatureLoss(torch.nn.Module):
    '''Given a content style reference images will find the style and content loss'''

    def __init__(self,
                 style_img,
                 style_weight,
                 content_weight
                 ) -> None:
        super(FeatureLoss, self).__init__()

        self.device = style_img.device

        self.style_weight = style_weight
        self.content_weight = content_weight

        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        normalization_mean_default = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        normalization_std_default = torch.tensor([0.229, 0.224, 0.225], device=self.device)

        # Load the VGG
        # model_file = 'vgg19.pt'
        # if Path(model_file).is_file():
        #     print(f'Loading model {model_file}')
        #     cnn = torch.load(model_file)
        # else:
        cnn = models.vgg19(pretrained=True).to(self.device)
        # torch.save(cnn, model_file)

        self.cnn = cnn.features.eval()

        self.style_model, self.style_losses, self.content_features = self.get_style_model_and_losses(
            self.cnn,
            style_img,
            normalization_mean=normalization_mean_default,
            normalization_std=normalization_std_default,
            content_layers=content_layers_default,
            style_layers=style_layers_default
        )

    def forward(self, input_img, content_img):
        # numel_input = torch.numel(input_img)
        # numel_content = torch.numel(content_img)
        # assert numel_input == numel_content, 'Input image and content image must be in equal sizes!'
        # dim = int(math.sqrt(numel_input / 3))
        #
        # input_img = input_img.view(dim, dim, 3).permute(2, 0, 1).unsqueeze(dim=0)
        # content_img = content_img.view(dim, dim, 3).permute(2, 0, 1).unsqueeze(dim=0)

        # collect feature loss tensors for the target/content image
        self.style_model(content_img)
        target_content_features = [cf.content_feature for cf in self.content_features]

        # another forward pass for the input image
        self.style_model(input_img)

        style_score = 0
        content_score = 0

        # style loss
        for sl in self.style_losses:
            style_score += sl.loss

        # content loss
        input_content_features = [cf.content_feature for cf in self.content_features]
        for in_cl_feat, target_cl_feat in zip(input_content_features, target_content_features):
            content_score += F.mse_loss(in_cl_feat, target_cl_feat)

        # weight the loss and combine
        style_score *= self.style_weight
        content_score *= self.content_weight

        loss = style_score + content_score

        return loss

    def get_style_model_and_losses(self,
                                   cnn,
                                   style_img,
                                   normalization_mean,
                                   normalization_std,
                                   content_layers,
                                   style_layers
                                   ):
        '''Build model to be used for style/content loss'''

        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(
            normalization_mean,
            normalization_std
        )

        # to have iterable access to a list of content features and style losses
        content_features = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            # add content loss to the model:
            if name in content_layers:
                # Get the feature map of the content image using the half built model
                content_feature = ContentFeature()
                model.add_module("content_feat_{}".format(i), content_feature)
                content_features.append(content_feature)

            # add style loss to the model:
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentFeature) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_features


class ContentFeature(torch.nn.Module):
    '''Extract the feature map'''

    def __init__(self):
        super(ContentFeature, self).__init__()

    def forward(self, input):
        # Save the content feature
        self.content_feature = input
        return input


class StyleLoss(torch.nn.Module):
    '''Compute the style loss using Gram matrices'''

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()

    def forward(self, input):
        if self.target.device.type == 'cpu':
            # TODO avoid doing this
            self.target = self.target.to(self.device)

        G = self._gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def _gram_matrix(self, input):
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        a, b, c, d = input.size()

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # 'normalize' the gram matrix by dividing by size of the feature map
        return G.div(a * b * c * d)


class Normalization(torch.nn.Module):
    '''VGG Normalisation (pretrained models expect this)'''

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.

        self.device = torch.device('cuda')

        self.mean = mean.view(-1, 1, 1).to(self.device)
        self.std = std.view(-1, 1, 1).to(self.device)

    def forward(self, img):
        # normalize img
        if self.mean.device.type == 'cpu':
            # TODO avoid doing this
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

        return (img - self.mean) / self.std


loss_dict = {'nerfw': MSELoss,
             'style': FeatureLoss}
