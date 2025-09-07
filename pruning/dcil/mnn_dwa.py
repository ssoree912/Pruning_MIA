import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################################################
# DWA (Dynamic Weight Adjustment) Maskers for 3 experiments
################################################################

class MaskerScalingReactivationOnly(torch.autograd.Function):
    """Reactivation-only: g' = g*m + alpha*g*(1-m)*f where f = ||w| - tau|"""
    @staticmethod
    def forward(ctx, x, mask, alpha, threshold):
        ctx.save_for_backward(mask, x, threshold)
        ctx.alpha = alpha
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x, threshold = ctx.saved_tensors
        alpha = ctx.alpha
        f = torch.abs(torch.abs(x) - threshold)  # f = ||w| - tau|
        g_new = (grad_out * mask) + (grad_out * (1 - mask) * f * alpha)
        return g_new, None, None, None


class MaskerScalingKillActivePlainDead(torch.autograd.Function):
    """Kill-active: g' = beta*g*m*|w| + g*(1-m)"""
    @staticmethod
    def forward(ctx, x, mask, beta):
        ctx.save_for_backward(mask, x)
        ctx.beta = beta
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x = ctx.saved_tensors
        beta = ctx.beta
        b = torch.abs(x)  # |w|
        g_new = (grad_out * mask * b * beta) + (grad_out * (1 - mask))
        return g_new, None, None


class MaskerScalingKillAndReactivate(torch.autograd.Function):
    """Kill & Reactivate: g' = beta*g*m*|w| + alpha*g*(1-m)*f"""
    @staticmethod
    def forward(ctx, x, mask, alpha, beta, threshold):
        ctx.save_for_backward(mask, x, threshold)
        ctx.alpha = alpha
        ctx.beta = beta
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x, threshold = ctx.saved_tensors
        alpha, beta = ctx.alpha, ctx.beta
        f = torch.abs(torch.abs(x) - threshold)  # f = ||w| - tau|
        b = torch.abs(x)  # |w|
        g_new = (grad_out * mask * b * beta) + (grad_out * (1 - mask) * f * alpha)
        return g_new, None, None, None, None


################################################################
# MaskConv2d: DWA 3가지 모드 지원 버전
################################################################
class MaskConv2dDWA(nn.Conv2d):
    """
    DWA 실험용 MaskConv2d
    forward_type 선택:
      - "reactivate_only"        : (1) g*m + alpha*g*(1-m)*f
      - "kill_active_plain_dead" : (2) beta*g*m*|w| + g*(1-m)
      - "kill_and_reactivate"    : (3) beta*g*m*|w| + alpha*g*(1-m)*f
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode
        )
        
        # DWA 실험 설정
        self.forward_type = "reactivate_only"  # 기본값
        
        # pruning mask & hyper-params
        self.mask = Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.threshold = Parameter(torch.tensor(0.05), requires_grad=False)  # 초기 threshold
        self.alpha = 1.0  # reactivation 강도
        self.beta = 1.0   # kill 강도
        
    def update_threshold(self, percentile=50):
        """threshold를 현재 가중치의 percentile로 동적 업데이트"""
        with torch.no_grad():
            weight_abs = torch.abs(self.weight)
            self.threshold.data = torch.quantile(weight_abs, percentile / 100.0)

    def forward(self, x):
        ft = self.forward_type.lower()
        
        if ft == "reactivate_only":
            masked_w = MaskerScalingReactivationOnly.apply(
                self.weight, self.mask, self.alpha, self.threshold
            )
        elif ft == "kill_active_plain_dead":
            masked_w = MaskerScalingKillActivePlainDead.apply(
                self.weight, self.mask, self.beta
            )
        elif ft == "kill_and_reactivate":
            masked_w = MaskerScalingKillAndReactivate.apply(
                self.weight, self.mask, self.alpha, self.beta, self.threshold
            )
        else:
            raise NotImplementedError(f"Unknown forward_type: {self.forward_type}")

        # 표준 Conv 연산
        return F.conv2d(x, masked_w, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)