"""
Unified utilities facade aggregating helpers from:
- utils_wemem.py (model factory, optimizers, losses)
- utils.py (training helpers, schedulers, meters)
- common_utils.py (lightweight meters, logger helpers)

Import from here to avoid scattered imports across the codebase.
"""
from typing import Any

# Pull selected symbols from utils_wemem
from typing import Callable, Optional
import torch
import torch.nn.functional as F
import numpy as np
import random

# Native implementations (so we can drop utils_wemem)
def weight_init(m):
    from torch.nn import init
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()

def get_optimizer(optimizer_name, parameters, lr, weight_decay=0):
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_name == "":
        return None
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def CrossEntropy_soft(input, target, reduction='mean'):
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)

def one_hot_embedding(y, num_classes=10, dtype=torch.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype).to(y.device)
    return torch.scatter(zeros, scatter_dim, y_tensor, 1)

# Training utilities (ported from legacy utils.py)
from copy import deepcopy
import csv, shutil
import pathlib
from os import remove
from os.path import isfile


class TrainAverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


class TrainProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters, self.prefix = meters, prefix
    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)] + [str(m) for m in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)); fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ScoreMeter(object):
    def __init__(self):
        self.label = []
        self.prediction = []
        self.score = None
    def update(self, output, target):
        pred = torch.argmax(output, dim=-1)
        self.prediction += pred.detach().cpu().tolist()
        self.label += target.detach().cpu().tolist()


def set_scheduler(optimizer, args):
    import torch.optim.lr_scheduler as lrs
    if getattr(args, 'scheduler', 'multistep') == 'step':
        return lrs.StepLR(optimizer, step_size=getattr(args, 'step_size', 30), gamma=getattr(args, 'gamma', 0.1))
    elif args.scheduler == 'multistep':
        if int(getattr(args, 'warmup_lr_epoch', 0)) > 0:
            return GradualWarmupScheduler(
                optimizer,
                warmup_epoch=args.warmup_lr_epoch,
                milestones=getattr(args, 'milestones', [100,150]),
                init_lr=getattr(args, 'lr', 0.1),
                gamma=getattr(args, 'gamma', 0.1),
                total_epoch=getattr(args, 'epochs', 200),
                warmup_init_lr=getattr(args, 'warmup_lr', 0.1)
            )
        else:
            return lrs.MultiStepLR(optimizer, milestones=getattr(args, 'milestones', [100,150]), gamma=getattr(args, 'gamma', 0.1))
    elif args.scheduler == 'exp':
        return lrs.ExponentialLR(optimizer, gamma=getattr(args, 'gamma', 0.1))
    elif args.scheduler == 'cosine':
        return lrs.CosineAnnealingLR(optimizer, T_max=getattr(args, 'step_size', 30))
    else:
        raise ValueError(f"Unsupported scheduler: {getattr(args, 'scheduler', None)}")


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_arch_name(args):
    arch_name = deepcopy(args.arch)
    if args.arch in ['resnet']:
        arch_name += str(args.layers)
    elif args.arch in ['wideresnet']:
        arch_name += f"{args.layers}_{int(args.width_mult)}"
    return arch_name


def save_ckpt(arch_name, dataset, state, ckpt_name='ckpt_best.pth'):
    dir_ckpt = pathlib.Path('checkpoint')
    dir_path = dir_ckpt / arch_name / dataset
    dir_path.mkdir(parents=True, exist_ok=True)
    if ckpt_name is None:
        ckpt_name = 'ckpt_best.pth'
    model_file = dir_path / ckpt_name
    torch.save(state, model_file)


def save_summary(arch_name, dataset, name, summary):
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)
    file_name = f'{arch_name}_{dataset}_{name}.csv'
    file_summ = dir_path / file_name
    if summary[0] == 0:
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['Epoch', 'Acc@1_train', 'Acc@5_train', 'Acc@1_valid', 'Acc@5_valid']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = dir_path / 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in); writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


def save_eval(summary):
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)
    file_summ = dir_path / 'eval.csv'
    if not isfile(file_summ):
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['ckpt', 'Acc@1', 'Acc@5']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = dir_path / 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in); writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


class GradualWarmupScheduler(object):
    def __init__(self, optimizer, warmup_epoch, milestones, init_lr, gamma, total_epoch, warmup_init_lr=0.1):
        self.optimizer = optimizer
        self.milestones = [0, warmup_epoch] + list(milestones) + [total_epoch]
        self.interval_epoch = [(self.milestones[i], self.milestones[i + 1]) for i in range(len(self.milestones) - 1)]
        self.cur_interval_idx = 0
        self.interval_lr = [(warmup_init_lr, init_lr)]
        for i in range(len(milestones) + 1):
            self.interval_lr.append(init_lr * (gamma ** i))
        self.total_epoch = total_epoch
        self.finished = False
        self.lr = warmup_init_lr if warmup_epoch > 0 else init_lr
    def step(self, epoch):
        # Determine interval
        while self.cur_interval_idx + 1 < len(self.milestones) and epoch >= self.milestones[self.cur_interval_idx + 1]:
            self.cur_interval_idx += 1
        # Compute LR for current interval (linear warmup else step)
        start_e, end_e = self.interval_epoch[self.cur_interval_idx]
        if self.cur_interval_idx == 0 and end_e > start_e:
            ratio = (epoch - start_e) / max(1, (end_e - start_e))
            lr = self.interval_lr[0][0] + ratio * (self.interval_lr[0][1] - self.interval_lr[0][0])
        else:
            lr = self.interval_lr[self.cur_interval_idx]
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.lr = lr

# Pull lightweight helpers from common_utils.py
# Lightweight logger helpers (inline minimal wrappers)
from torch.utils.tensorboard import SummaryWriter
import pathlib

class LightAverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"

class LightProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters, self.prefix = meters, prefix
    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)] + [str(m) for m in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)); fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# --- Backward-compatibility aliases ---
# Older runners expect AverageMeter/ProgressMeter symbols
AverageMeter = LightAverageMeter
ProgressMeter = LightProgressMeter

class SummaryLogger(SummaryWriter):
    def __init__(self, path):
        super().__init__()
        file_path = "./logs/" + path
        self.logger = SummaryWriter(file_path)
    def add_scalar_group(self, main_tag, tag_scalar_dict, global_step):
        for sub_tag, scalar in tag_scalar_dict.items():
            self.logger.add_scalar(main_tag + f"/{sub_tag}", scalar, global_step)
    def add_max_acc(self, main_tag, tag_scalar_dict):
        for sub_tag, scalar in tag_scalar_dict.items():
            var = main_tag.split("/")[0] + f"_{sub_tag}"
            if not hasattr(self, var):
                setattr(self, var, -1)
            cur = getattr(self, var)
            if cur <= scalar:
                self.logger.add_scalar(main_tag + f"/{sub_tag}", scalar, 0)
                setattr(self, var, scalar)

def save_model_simple(arch_name, dataset, state, ckpt_name="ckpt_best.pth"):
    dir_ckpt = pathlib.Path("checkpoint")
    dir_path = dir_ckpt / arch_name / dataset
    dir_path.mkdir(parents=True, exist_ok=True)
    if ckpt_name is None:
        ckpt_name = "ckpt_best.pth"
    model_file = dir_path / ckpt_name
    torch.save(state, model_file)


def seed_worker(worker_id: int) -> Any:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    # model/optim
    'get_optimizer', 'weight_init', 'CrossEntropy_soft', 'one_hot_embedding', 'seed_worker',
    # training utils
    'TrainAverageMeter', 'TrainProgressMeter', 'ScoreMeter', 'set_scheduler', 'accuracy', 'set_arch_name',
    'save_ckpt', 'save_summary', 'save_eval',
    # light utils
    'LightAverageMeter', 'LightProgressMeter', 'SummaryLogger', 'save_model_simple',
    # compat aliases
    'AverageMeter', 'ProgressMeter',
]
