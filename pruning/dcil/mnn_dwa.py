import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################################################
# 1) 기존 Static / Dynamic (DPF) 마스커
################################################################

class MaskerStatic(torch.autograd.Function):
    """Static pruning: dead weights stay dead (gradient masked by mask)"""
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None


class MaskerDynamic(torch.autograd.Function):
    """DPF: dead weights can reactivate (no gradient masking)"""
    @staticmethod
    def forward(ctx, x, mask):
        # dynamic: no need to save mask for backward since grad is unchanged
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


################################################################
# 2) DWA (Dynamic Weight Adjustment) – 3가지 실험용 마스커
################################################################

class MaskerScalingReactivationOnly(torch.autograd.Function):
    """
    (1) Reactivation-only: 죽인걸 살리는 것 
        g' = g*m + alpha * g * (1-m) * f
      where f = ||w| - tau|
    """
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
    """
    (2) Kill-active & Plain-dead: 살아있는걸 죽이는것만 mask = 1  
        g' = beta * g * m * |w| + g * (1-m)
      활성 가중치는 |w|로 스케일해 '죽이는' 방향으로, 비활성은 평범한 grad (재활성화 효과 없음)
    """
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
    """
    (3) Kill & Reactivate (양쪽 모두):
        g' = beta * g * m * |w| + alpha * g * (1-m) * f
      where f = ||w| - tau|
    """
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
# 3) 통합 MaskConv2d
#    - legacy: type_value 사용 (0 기본, 5 Static, 6 Dynamic)
#    - DWA   : forward_type 사용 ('reactivate_only', 'kill_active_plain_dead', 'kill_and_reactivate')
################################################################

class MaskConv2d(nn.Conv2d):
    """
    통합 MaskConv2d
    - legacy 모드 (기본): self.forward_type is None -> self.type_value에 따라 동작
        * type_value == 0 : weight * mask (출력만 sparse)
        * type_value == 5 : MaskerStatic (gradient도 mask로 차단)
        * type_value == 6 : MaskerDynamic (gradient 그대로 통과)
    - DWA 모드: self.forward_type in {'reactivate_only', 'kill_active_plain_dead', 'kill_and_reactivate'}
        * 아래 3가지 마스커를 사용
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        # 공통: pruning mask (학습 X)
        self.mask = Parameter(torch.ones_like(self.weight), requires_grad=False)

        # legacy 경로
        self.type_value = 0  # 0: 기본 sparse 출력, 5: Static, 6: Dynamic

        # DWA 경로
        self.forward_type = None  # None => legacy, 문자열 모드면 DWA 사용
        self.alpha = 1.0
        self.beta = 1.0
        self.threshold = Parameter(torch.tensor(0.05, dtype=self.weight.dtype, device=self.weight.device),
                                   requires_grad=False)

    # 편의 메서드: DWA threshold 업데이트(가중치 절대값의 p-분위수)
    def update_threshold(self, percentile: int = 50):
        with torch.no_grad():
            weight_abs = torch.abs(self.weight)
            self.threshold.data = torch.quantile(weight_abs, percentile / 100.0)

    def forward(self, x):
        # 1) DWA 모드가 설정되어 있으면 우선 적용
        if isinstance(self.forward_type, str):
            ft = self.forward_type.lower()
            if ft == "reactivate_only":
                masked_weight = MaskerScalingReactivationOnly.apply(
                    self.weight, self.mask, self.alpha, self.threshold
                )
            elif ft == "kill_active_plain_dead":
                masked_weight = MaskerScalingKillActivePlainDead.apply(
                    self.weight, self.mask, self.beta
                )
            elif ft == "kill_and_reactivate":
                masked_weight = MaskerScalingKillAndReactivate.apply(
                    self.weight, self.mask, self.alpha, self.beta, self.threshold
                )
            else:
                raise NotImplementedError(f"Unknown forward_type: {self.forward_type}")

        # 2) legacy 경로(type_value)
        else:
            if self.type_value == 5:
                masked_weight = MaskerStatic.apply(self.weight, self.mask)
            elif self.type_value == 6:
                masked_weight = MaskerDynamic.apply(self.weight, self.mask)
            else:
                # Default: sparse output only
                masked_weight = self.weight * self.mask

        # 표준 Conv
        return F.conv2d(x, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# 호환성: 이전 코드가 mnn_dwa.MaskConv2dDWA 를 찾는 경우를 대비
MaskConv2dDWA = MaskConv2d