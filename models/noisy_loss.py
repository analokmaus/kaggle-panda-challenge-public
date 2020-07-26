import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


'''
Basic loss definition
'''

class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def linear_combination(self, x, y, epsilon):
        return epsilon*x + (1-epsilon)*y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target, reduction='mean'):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), reduction=reduction)
        nll = F.nll_loss(log_preds, target, reduction=reduction)
        return self.linear_combination(loss/n, nll, self.epsilon)


def soft_cross_entropy_loss(logits, targets, weights=1, reduction='mean'):
    if len(targets.shape) == 1 or targets.shape[1] == 1:
        onehot_targets = torch.eye(logits.shape[1])[targets].to(logits.device)
    else:
        onehot_targets = targets
    loss = -torch.sum(onehot_targets * F.log_softmax(logits, 1), 1)
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()


def coral_loss(logits, targets, weights=1, reduction='mean'):
    loss = (-torch.sum(
        (F.logsigmoid(logits)*targets + 
        (F.logsigmoid(logits) - logits) * (1 - targets)) * weights,
        dim=1))
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()


def get_loss(name):
    loss_dict = {
        'CrossEntropy': F.cross_entropy,
        'SoftCrossEntropy': soft_cross_entropy_loss,
        'LabelSmoothingCrossEntropy':
        LabelSmoothingCrossEntropy(epsilon=0.1),
        'SCE': SymmetricCrossEntropy(), 
        'BCEWithLogitsLoss': F.binary_cross_entropy_with_logits,
        'Coral': coral_loss, 
        'MSELoss': F.mse_loss,
        'MAELoss': F.l1_loss,
        'huber': F.smooth_l1_loss,
    }
    return loss_dict[name]


'''
Common
'''

def _check_input_type(x, y, loss):
    if loss in ['MSELoss', 'MAELoss', 'huber']:
        if x.shape[-1] == 1:
            return x.squeeze(), y.float()
        else:
            return x, y.float()
    elif loss in ['CrossEntropy']:
        return x, y.long()
    else:
        return x, y


'''
 Loss functions
'''

class OUSMLoss(nn.Module):
    '''
    Implementation of 
    Loss with Online Uncertainty Sample Mining:
    https://arxiv.org/pdf/1901.07759.pdf

    # Params
    k: num of samples to drop in a mini batch
    loss: loss function name (see get_loss function above)
    trigger: the epoch it starts to train on OUSM (please call `.update(epoch)` each epoch)
    '''

    def __init__(self, k=1, loss='MSELoss', trigger=5, ousm=False):
        super(OUSMLoss, self).__init__()
        self.k = k
        self.loss_name = loss
        self.loss = get_loss(loss)
        self.trigger = trigger
        self.ousm = ousm

    def forward(self, logits, targets, indices=None):
        logits, targets = _check_input_type(logits, targets, self.loss_name)
        bs = logits.shape[0]
        if self.ousm and bs - self.k > 0:
            losses = self.loss(logits, targets, reduction='none')
            if len(losses.shape) == 2:
                losses = losses.mean(1)
            _, idxs = losses.topk(bs-self.k, largest=False)
            losses = losses.index_select(0, idxs)
            return losses.mean()
        else:
            return self.loss(logits, targets)

    def update(self, current_epoch):
        self.current_epoch = current_epoch
        if current_epoch == self.trigger:
            self.ousm = True
            print('criterion: ousm is True.')

    def __repr__(self):
        return f'OUSM(loss={self.loss_name}, k={self.k}, trigger={self.trigger}, ousm={self.ousm})'
         

class JointOptimizationLoss(nn.Module):
    '''
    Reimplementation of
    Joint Optimization Framework for Learning with Noisy Labels:
    https://arxiv.org/pdf/1803.11364.pdf

    # Params:
    alpha, beta: see original paper
    window: window size to calculate pseudo labels
    trigger: (the epoch to start to save pseudo labels, 
              the epoch to start to train on pseudo labels)
    '''

    def __init__(self, alpha=1.0, beta=0.5, trigger=(1, 20), window=5, 
                 debug=False, save_pseudo_targets=None):
        super(JointOptimizationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.trigger = trigger
        self.window = window
        self.debug = debug
        self.update_target = False
        self.pseudo_target = False
        self.current_epoch = 0
        self.save_pseudo_targets = save_pseudo_targets

    def forward(self, logits, targets, indices):
        device = logits.device
        bs, classes = logits.shape
        if logits.requires_grad: # training
            if self.pseudo_target:
                targets = self.pseudo_targets[indices].to(device)
                return soft_cross_entropy_loss(logits, targets)
            else:
                targets = self.soft_targets[indices].to(device)
                p = torch.ones(classes).to(device) / classes
                probas = logits.softmax(1)
                avg_proba = probas.mean(0)
                L_c = soft_cross_entropy_loss(logits, targets)
                L_p = -torch.sum(torch.log(avg_proba) * p)
                L_e = -torch.mean(torch.sum(probas * \
                                          F.log_softmax(logits, 1), 1))
                loss = L_c + self.alpha * L_p + self.beta * L_e
                if self.update_target:
                    count = self.current_epoch % self.window
                    self.pseudo_targets[count, indices] = probas.detach().cpu()
                return loss
        else:
            return soft_cross_entropy_loss(logits, targets)
    
    def init_targets(self, targets):
        # Both on CPU
        self.soft_targets = torch.eye(6)[torch.from_numpy(targets)].float()
        self.pseudo_targets = torch.zeros((self.window, len(targets), 6)).float()
        if self.debug:
            print('soft_targets, pseudo_targets: ', 
                  self.soft_targets.shape, self.pseudo_targets.shape)

    def update_pseudo_targets(self):
        self.pseudo_targets = self.pseudo_targets.mean(0)
        if self.save_pseudo_targets:
            torch.save(self.pseudo_targets, self.save_pseudo_targets)

    def load_pseudo_targets(self):
        self.pseudo_targets = torch.load(self.save_pseudo_targets)

    def update(self, current_epoch):
        self.current_epoch = current_epoch
        if current_epoch == self.trigger[0]:
            self.update_target = True
            print('criterion: update target is True.')
        if current_epoch == self.trigger[1]:
            self.update_target = False
            self.pseudo_target = True
            self.update_pseudo_targets()
            print('criterion: update_target is False. pseudo_target is True.')

    def __repr__(self):
        return f'JointOptimizationLoss(alpha={self.alpha}, beta={self.beta}, trigger={self.trigger})'


class SymmetricCrossEntropy(nn.Module):
    '''
    Reimplementation of 
    Symmetric Loss:
    https://arxiv.org/pdf/1908.06112.pdf
    '''

    def __init__(self, alpha=0.1, beta=1.0):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, reduction='mean'):
        logits, targets = _check_input_type(logits, targets, 'CrossEntropy')
        device = logits.device
        onehot_targets = torch.eye(6)[targets].to(device)
        ce_loss = F.cross_entropy(logits, targets, reduction=reduction)
        rce_loss = (-onehot_targets*logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == 'mean':
            rce_loss = rce_loss.mean()
        elif reduction == 'sum':
            rce_loss = rce_loss.sun()
        return self.alpha * ce_loss + self.beta * rce_loss

    def __repr__(self):
        return f'SymmetricCrossEntropy(alpha={self.alpha}, beta={self.beta})'


class IterativeSelfLearningLoss(nn.Module):
    '''
    Implementation of 
    Deep Self-Learning From Noisy Labels: 
    https://arxiv.org/pdf/1908.02160.pdf
    '''

    def __init__(self, model, alpha=0.5, loss='CrossEntropy'):
        super(IterativeSelfLearningLoss, self).__init__()
        self.alpha = alpha
        self.model = model
        self.loss_name = loss
        self.loss = get_loss(loss)
        
    def forward(self, logits, targets, indices=None):
        logits, targets = _check_input_type(logits, targets, self.loss_name)
        if logits.requires_grad:
            noisy_targets = targets
            corrected_targets = self.model.train_pseudo_labels[indices].to(targets.device)
        else:
            noisy_targets = targets
            corrected_targets = self.model.valid_pseudo_labels[indices].to(targets.device)
        
        _, noisy_targets = _check_input_type(
            logits, noisy_targets, self.loss_name)
        _, corrected_targets = _check_input_type(
            logits, corrected_targets, self.loss_name)
        noisy_loss = self.loss(logits, noisy_targets)
        corrected_loss = self.loss(logits, corrected_targets)
        return (1 - self.alpha) * noisy_loss + self.alpha * corrected_loss
