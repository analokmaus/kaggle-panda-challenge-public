import numpy as np
import torch
from functools import partial
import scipy as sp
from sklearn.metrics import cohen_kappa_score
from kuma_utils.metrics import QWK, Accuracy
from models.noisy_loss import get_loss, _check_input_type


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class OptimizedRounder:

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']


class CustomQWK:

    def __init__(self, num_classes=6, optim=False, loss=None, p=1.0):
        super().__init__()
        self.exclude_hard = False
        self.qwk = QWK(num_classes)
        self.p = p
        self.optim = optim
        self.optR = OptimizedRounder()
        if loss is not None:
            # evaluate on p % high certainty samples
            self.exclude_hard = True
            self.loss_name = loss
            self.loss = get_loss(loss)

    def __call__(self, approx, target):
        if self.exclude_hard:
            approx, target = _check_input_type(approx, target, self.loss_name)
            losses = self.loss(approx, target, reduction='none')
            if len(losses.shape) == 2:
                losses = losses.mean(1)
            _, use_idxs = losses.topk(round(approx.shape[0]*self.p), largest=False)
            _approx = approx.index_select(0, use_idxs)
            _target = target.index_select(0, use_idxs)
        else:
            _approx = approx
            _target = target

        if _approx.shape[1] == 5:
            _approx = _approx.sigmoid().sum(1)
            _target = _target.sum(1)
        elif _approx.shape[1] >= 6:
            _approx = _approx[:, :6].softmax(1) @ torch.from_numpy(np.arange(6)).float()
        else:
            pass

        if self.optim:
            self.optR.fit(_approx, _target)
            coef = self.optR.coefficients()
            print(coef)
            _approx = self.optR.predict(_approx, coef)

        return self.qwk(_target, _approx)

    def __repr__(self):
        return f'CustomQWK(p={self.p}, optim={self.optim})'


class CustomAccuracy:

    def __init__(self, num_classes=6):
        super().__init__()
        self.acc = Accuracy().torch

    def __call__(self, approx, target):
        if approx.shape[1] == 5:
            return self.acc(approx.sigmoid().sum(1).round(), target.sum(1).round())
        elif approx.shape[1] == 7:
            return self.acc(approx[:, :-1], target)
        else:
            return self.acc(approx, target)

    def __repr__(self):
        return f'CustomAccuracy'
