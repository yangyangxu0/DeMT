import torch
import torch.nn as nn

import torchvision
import numpy as np

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


def spatial_normalize_pred(pred, image, ignore_index=255):
    prob = {}
    for t in pred.keys():
        task_pred = pred[t]
        batch_size, num_classes, H, W = task_pred.size()
        # check for ignore_index in input image, arising for example from data augmentation
        ignore_mask = (nn.functional.interpolate(image, size=(
            H, W), mode='nearest') == ignore_index).any(dim=1, keepdim=True)
        # so they won't contribute to the softmax
        task_pred[ignore_mask.expand_as(task_pred)] = -float('inf')
        c_probs = nn.functional.softmax(
            task_pred.view(batch_size, num_classes, -1), dim=2)
        # if the whole input image consisted of ignore regions, then context probs should just be zero
        prob[t] = torch.where(torch.isnan(
            c_probs), torch.zeros_like(c_probs), c_probs)
    return prob





def prep_a_net(model_name, shall_pretrain):
    model = getattr(torchvision.models, model_name)(shall_pretrain)
    if "resnet" in model_name:
        model.last_layer_name = 'fc'
    elif "mobilenet_v2" in model_name:
        model.last_layer_name = 'classifier'
    return model

def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")

def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    if size == h:
        return im
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y : (y + size), x : (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop

def get_patch(images, action_sequence, patch_size):
    """Get small patch of the original image"""
    batch_size = images.size(0)
    image_size = images.size(2)

    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()
    patches = []
    for i in range(batch_size):
        per_patch = images[i, :,
                    (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size).item()),
                    (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size).item())]

        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

    return torch.cat(patches, 0)


