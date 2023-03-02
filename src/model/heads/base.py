import torch
import torch.nn as nn


class BaseHead(nn.Module):
    def __init__(self, tasks, task_channel_mapping, in_index, idx_to_planes):
        super().__init__()
        self.tasks = tasks
        self.task_channel_mapping = task_channel_mapping
        self.in_index = in_index
        self.idx_to_planes = idx_to_planes
        self.in_channels = sum([self.idx_to_planes[i] for i in self.in_index])

    def forward(self, inp, inp_shape):
        raise NotImplementedError

    def _transform_inputs(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        upsampled_inputs = [
            nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=False) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        return inputs

    def init_weights(self):
        # By default we use pytorch default initialization. Heads can have their own init.
        # Except if `logits` is in the name, we override.
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if 'logits' in name:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
