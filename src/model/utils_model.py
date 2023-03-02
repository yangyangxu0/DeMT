import os
import numpy as np
from PIL import Image
import torch


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


@torch.no_grad()
def save_predictions(task, preds, meta, save_dir):
    if task in ['edge', 'sal']:
        preds = 255 * torch.sigmoid(preds.squeeze(1))
    elif task in ['semseg', 'human_parts']:
        preds = torch.argmax(preds, dim=1)
    elif task == 'normals':
        norm = torch.norm(preds, p='fro', dim=1, keepdim=True).expand_as(preds)
        preds = 255 * (preds.div(norm) + 1.0) / 2.0
        preds[norm == 0] = 0
    elif task == 'depth':
        pass
    else:
        raise ValueError

    for idx, pred in enumerate(preds):
        im_height = meta['im_size'][0][idx]
        im_width = meta['im_size'][1][idx]
        im_name = meta['image'][idx]

        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[-2:]:
            delta_height = max(pred.shape[-2] - im_height, 0)
            delta_width = max(pred.shape[-1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                height_location = [delta_height // 2,
                                   (delta_height // 2) + im_height]
                width_location = [delta_width // 2,
                                  (delta_width // 2) + im_width]
                pred = pred[..., height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[-2:] == (im_height, im_width)
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
        arr = pred.cpu().numpy()
        if task == 'depth':
            np.save(os.path.join(save_dir, '{}.npy'.format(im_name)), arr)
        else:
            image = Image.fromarray(arr.astype(np.uint8))
            image.save(os.path.join(save_dir, '{}.png'.format(im_name)))
