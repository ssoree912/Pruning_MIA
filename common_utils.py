"""
Common utilities extracted from utils.py to avoid import conflicts
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import pathlib

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def set_scheduler(optimizer, args):
    """Sets the learning rate scheduler"""
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)
    
    return scheduler


def set_arch_name(args):
    """Set architecture name"""
    if hasattr(args, 'arch') and hasattr(args, 'layers'):
        return f"{args.arch}{args.layers}"
    else:
        return "model"

class SummaryLogger(SummaryWriter):
    def __init__(self, path):
        super().__init__()
        file_path = "./logs/" + path
        self.logger = SummaryWriter(file_path)

    def add_scalar_group(self, main_tag, tag_scalar_dict, global_step):
        for sub_tag, scalar in tag_scalar_dict.items():
            self.logger.add_scalar(
                main_tag + "/{}".format(sub_tag), scalar, global_step
            )

    def add_max_acc(self, main_tag, tag_scalar_dict):
        for sub_tag, scalar in tag_scalar_dict.items():
            variable_name = main_tag.split("/")[0] + "_{}".format(sub_tag)
            if not hasattr(self, variable_name):
                setattr(self, variable_name, -1)

            current_max_acc = getattr(self, variable_name)
            if current_max_acc <= scalar:
                self.logger.add_scalar(main_tag + "/{}".format(sub_tag), scalar, 0)
                setattr(self, variable_name, scalar)
def save_model(arch_name, dataset, state, ckpt_name="ckpt_best.pth"):
    r"""Save the model (checkpoint) at the training time"""
    dir_ckpt = pathlib.Path("checkpoint")
    dir_path = dir_ckpt / arch_name / dataset
    dir_path.mkdir(parents=True, exist_ok=True)

    if ckpt_name is None:
        ckpt_name = "ckpt_best.pth"
    model_file = dir_path / ckpt_name
    torch.save(state, model_file)