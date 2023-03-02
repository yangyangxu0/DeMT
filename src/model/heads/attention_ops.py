import torch

from .ops import locatt_ops


class similarFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.similar_forward(
            x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.similar_backward(
            x_loc, grad_outputs, kH, kW, True)
        grad_loc = locatt_ops.localattention.similar_backward(
            x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.weighting_forward(
            x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.weighting_backward_ori(
            x_weight, grad_outputs, kH, kW)
        grad_weight = locatt_ops.localattention.weighting_backward_weight(
            x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None
