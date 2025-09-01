import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class MaskerStatic(torch.autograd.Function):
    """Static pruning: dead weights stay dead (gradient masking)"""
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None


class MaskerDynamic(torch.autograd.Function):
    """DPF: dead weights can reactivate (full gradient)"""
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None




class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)

        # 0 -> part use, 1-> full use
        self.type_value = 0

    def forward(self, input):
        if self.type_value == 5:
            masked_weight = MaskerStatic.apply(self.weight, self.mask)
        elif self.type_value == 6:
            masked_weight = MaskerDynamic.apply(self.weight, self.mask)
        else:
            # Default: sparse output (type_value=0)
            masked_weight = self.weight * self.mask

        # PyTorch version compatibility
        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)