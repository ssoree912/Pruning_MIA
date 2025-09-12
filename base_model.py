"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils.utils import get_model, get_optimizer, weight_init, CrossEntropy_soft, one_hot_embedding



class BaseModel:
    def __init__(self, model_type, device="cuda", save_folder="", num_cls=10,
                 optimizer="", lr=0.01, weight_decay=0, input_dim=100, epochs=0, attack_model_type='', weight_decay_mem=5e-4):
        self.model = get_model(model_type, num_cls, input_dim)
        self.model.to(device)
        self.model.apply(weight_init)
        self.device = device
        self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr, weight_decay)
        self.optimizer_risk = get_optimizer(optimizer, self.model.parameters(), lr, weight_decay_mem)
        if epochs == 0:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        self.softmax = nn.Softmax(dim=1)
        self.save_pref = save_folder
        self.num_cls = num_cls

        if attack_model_type:
            self.attack_model = get_model(attack_model_type, num_cls*2, 2)
            self.attack_model.to(device)
            self.attack_model.apply(weight_init)
            self.attack_model_optim = get_optimizer("adam", self.attack_model.parameters(), lr=0.001, weight_decay=5e-4)

    def _safe_forward(self, inputs):
        """Forward wrapper that adapts to models requiring additional args (e.g., type_value).
        Tries a plain call first, then falls back to common DWA/MaskConv variants.
        """
        # If a preferred type_value is set, try it first
        tv_pref = getattr(self, 'preferred_type_value', None)
        if tv_pref is not None:
            try:
                return self.model(inputs, type_value=tv_pref)
            except Exception:
                pass
        try:
            return self.model(inputs)
        except TypeError as e:
            msg = str(e)
            if 'type_value' in msg:
                # Try preferred first then common fallbacks
                tried = set([tv_pref] if tv_pref is not None else [])
                for tv in (0, 5, 6):
                    if tv in tried:
                        continue
                    try:
                        return self.model(inputs, type_value=tv)
                    except Exception:
                        continue
                # As a last resort, try keyword without value to surface clearer error
                return self.model(inputs, type_value=0)
            raise

    def train(self, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def train_defend_ml2(self, train_loader_risk=None, train_loader_gen=None, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        if train_loader_risk == None:
            for inputs_gen, targets_gen in train_loader_gen:
                inputs_gen, targets_gen = inputs_gen.to(self.device), targets_gen.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs_gen)
                loss = self.criterion(outputs, targets_gen)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * targets_gen.size(0)
                total += targets_gen.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets_gen).sum().item()
        elif train_loader_gen == None:
            for inputs_risk, targets_risk in train_loader_risk:
                inputs_risk, targets_risk = inputs_risk.to(self.device), targets_risk.to(self.device)
                self.optimizer_risk.zero_grad()
                outputs = self.model(inputs_risk)
                loss = self.criterion(outputs, targets_risk)
                loss.backward()
                self.optimizer_risk.step()
                total_loss += loss.item() * targets_risk.size(0)
                total += targets_risk.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets_risk).sum().item()
        else:
            for inputs_risk, targets_risk in train_loader_risk:
                inputs_risk, targets_risk = inputs_risk.to(self.device), targets_risk.to(self.device)
                self.optimizer_risk.zero_grad()
                outputs = self.model(inputs_risk)
                loss = self.criterion(outputs, targets_risk)
                loss.backward()
                self.optimizer_risk.step()
                total_loss += loss.item() * targets_risk.size(0)
                total += targets_risk.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets_risk).sum().item()
            for inputs_gen, targets_gen in train_loader_gen:
                inputs_gen, targets_gen = inputs_gen.to(self.device), targets_gen.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs_gen)
                loss = self.criterion(outputs, targets_gen)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * targets_gen.size(0)
                total += targets_gen.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets_gen).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: total Accuracy {:.3f}, total Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def train_defend_ppb(self, train_loader, log_pref="",defend_arg=None):
        self.model.train()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss1 = self.criterion(outputs, targets)
            ranked_outputs, _ = torch.topk(outputs, self.num_cls, dim=-1)
            size = targets.size(0)
            even_size = size // 2 * 2
            if even_size > 0:
                loss2 = F.kl_div(F.log_softmax(ranked_outputs[:even_size // 2], dim=-1),
                                 F.softmax(ranked_outputs[even_size // 2:even_size], dim=-1),
                                 reduction='batchmean')
            else:
                loss2 = torch.zeros(1).to(self.device)
            loss = loss1 + defend_arg * loss2
            total_loss += loss.item() * size
            total_loss1 += loss1.item() * size
            total_loss2 += loss2.item() * size
            total += size
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            loss.backward()
            self.optimizer.step()
        acc = 100. * correct / total
        total_loss /= total
        total_loss1 /= total
        total_loss2 /= total

        if self.scheduler:
            self.scheduler.step()
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}, Loss1 {:.3f}, Loss2 {:.3f}".format(
                log_pref, acc, total_loss, total_loss1, total_loss2))
        return acc, total_loss

    def train_defend_adv(self, train_loader, test_loader, log_pref="", privacy_theta=1.0):
        """
        modified from
        https://github.com/Lab41/cyphercat/blob/master/Defenses/Adversarial_Regularization.ipynb
        """
        total_loss = 0
        correct = 0
        total = 0
        infer_iterations = 7
        # train adversarial network

        train_iter = iter(train_loader)
        test_iter = iter(test_loader)
        train_iter2 = iter(train_loader)

        self.model.eval()
        self.attack_model.train()
        for infer_iter in range(infer_iterations):
            with torch.no_grad():
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                in_predicts = F.softmax(self.model(inputs), dim=-1)
                in_targets = F.one_hot(targets, num_classes=self.num_cls).float()

                try:
                    inputs, targets = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    inputs, targets = next(test_iter)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                out_predicts = F.softmax(self.model(inputs), dim=-1)
                out_targets = F.one_hot(targets, num_classes=self.num_cls).float()

                infer_train_data = torch.cat([torch.cat([in_predicts, in_targets], dim=-1),
                                              torch.cat([out_predicts, out_targets], dim=-1)], dim=0)
                infer_train_label = torch.cat([torch.ones(in_predicts.size(0)),
                                               torch.zeros(out_predicts.size(0))]).long().to(self.device)

            self.attack_model_optim.zero_grad()
            infer_loss = privacy_theta * F.cross_entropy(self.attack_model(infer_train_data), infer_train_label)
            infer_loss.backward()
            self.attack_model_optim.step()

        self.model.train()
        self.attack_model.eval()
        try:
            inputs, targets = next(train_iter2)
        except StopIteration:
            train_iter2 = iter(train_loader)
            inputs, targets = next(train_iter2)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss1 = self.criterion(outputs, targets)
        in_predicts = F.softmax(outputs, dim=-1)
        in_targets = F.one_hot(targets, num_classes=self.num_cls).float()
        infer_data = torch.cat([in_predicts, in_targets], dim=-1)
        infer_labels = torch.ones(targets.size(0)).long().to(self.device)
        infer_loss = F.cross_entropy(self.attack_model(infer_data), infer_labels)
        loss = loss1 - privacy_theta * infer_loss
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def train_defend_relaxloss(self, train_loader, epoch, log_pref="", alpha=1.0, upper=1.0):
        """
        modified from
        https://github.com/DingfanChen/RelaxLoss
        """
        self.model.train()
        total_loss = 0
        predict_correct = 0
        total = 0
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss_ce_full = self.crossentropy_noreduce(outputs, targets)
            loss_ce = torch.mean(loss_ce_full)
            size = targets.size(0)
            if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
                loss = (loss_ce - alpha).abs()
            else:
                if loss_ce > alpha:  # normal gradient descent
                    loss = loss_ce
                else:  # posterior flattening
                    pred = torch.argmax(outputs, dim=1)
                    correct = torch.eq(pred, targets).float()
                    confidence_target = self.softmax(outputs)[torch.arange(targets.size(0)), targets]
                    confidence_target = torch.clamp(confidence_target, min=0., max=upper)
                    confidence_else = (1.0 - confidence_target) / (self.num_cls - 1)
                    onehot = one_hot_embedding(targets, num_classes=self.num_cls)
                    soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_cls) \
                                   + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.num_cls)
                    loss = (1 - correct) * self.crossentropy_soft(outputs, soft_targets) - 1. * loss_ce_full
                    loss = torch.mean(loss)
            total_loss += loss.item() * size
            total += size
            _, predicted = outputs.max(1)
            predict_correct += predicted.eq(targets).sum().item()
            loss.backward()
            self.optimizer.step()
        acc = 100. * predict_correct / total
        total_loss /= total

        if self.scheduler:
            self.scheduler.step()
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss


    def test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._safe_forward(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                if isinstance(self.criterion, nn.BCELoss):
                    correct += torch.sum(torch.round(outputs) == targets)
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total 
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))            
        return acc, total_loss

    def save(self, epoch, acc, loss):
        save_path = f"{self.save_pref}/{epoch}.pth"
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'loss': loss,
            'state': self.model.state_dict()
        }
        torch.save(state, save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        # Support both metadata checkpoints and raw state_dict files
        model_state = None
        acc = None
        loss = None
        epoch = None
        if isinstance(state, dict) and 'state' in state and isinstance(state['state'], dict):
            model_state = state['state']
            acc = state.get('acc', None)
            loss = state.get('loss', None)
            epoch = state.get('epoch', None)
        elif isinstance(state, dict):
            # Likely a raw state_dict
            model_state = state
        else:
            # Unexpected format; try to use as state_dict directly
            model_state = state

        if verbose:
            print(f"Load model from {load_path}")
            if epoch is not None and acc is not None and loss is not None:
                print(f"Epoch {epoch}, Acc: {acc:.3f}, Loss: {loss:.3f}")

        try:
            self.model.load_state_dict(model_state)
            return acc if acc is not None else 0.0
        except Exception as e:
            # Retry with common adaptations
            from collections import OrderedDict
            target_state = self.model.state_dict()

            # 1) Strip 'module.' prefix if present
            new_state = OrderedDict()
            for k, v in model_state.items():
                nk = k[7:] if k.startswith('module.') else k
                new_state[nk] = v
            model_state = new_state

            # 2) Filter to matching keys and shapes
            filtered_state = OrderedDict()
            for k, v in model_state.items():
                if k in target_state and hasattr(v, 'shape') and hasattr(target_state[k], 'shape'):
                    if v.shape == target_state[k].shape:
                        filtered_state[k] = v

            missing = [k for k in target_state.keys() if k not in filtered_state]
            unexpected = [k for k in model_state.keys() if k not in filtered_state]
            if verbose:
                print("Warning: relaxed loading due to state_dict mismatch.")
                print(f"  Loaded params: {len(filtered_state)} | Missing: {len(missing)} | Skipped unexpected: {len(unexpected)}")

            self.model.load_state_dict(filtered_state, strict=False)
            return acc if acc is not None else 0.0

    def predict_target_sensitivity(self, data_loader, m=10, epsilon=1e-3):
        self.model.eval()
        predict_list = []
        sensitivity_list = []
        target_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self._safe_forward(inputs)
                predicts = F.softmax(outputs, dim=-1)
                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)

                if len(inputs.size()) == 4:
                    x = inputs.repeat((m, 1, 1, 1))
                elif len(inputs.size()) == 3:
                    x = inputs.repeat((m, 1, 1))
                elif len(inputs.size()) == 2:
                    x = inputs.repeat((m, 1))
                u = torch.randn_like(x)
                evaluation_points = x + epsilon * u
                new_predicts = F.softmax(self._safe_forward(evaluation_points), dim=-1)
                diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
                diff = diff.view(m, -1, self.num_cls)
                sensitivity = diff.mean(dim=0) / epsilon
                sensitivity_list.append(sensitivity.detach().data.cpu())

        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        sensitivities = torch.cat(sensitivity_list, dim=0)
        return predicts, targets, sensitivities
