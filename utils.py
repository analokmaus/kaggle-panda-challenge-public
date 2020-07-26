import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from kuma_utils.nn.training import *
from kuma_utils.metrics import QWK
from sklearn.metrics import confusion_matrix


class MyScheduler(_LRScheduler):

    def __init__(self, optimizer, config={10: 0.5, 20: 0.5, 30: 0.1}, last_epoch=-1):
            self.config = config
            super(MyScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        old_lr = [group['lr'] for group in self.optimizer.param_groups]
        if not self.last_epoch in self.config.keys():
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            new_lr = [group['lr'] * self.config[self.last_epoch]
                      for group in self.optimizer.param_groups]
            print(f'learning rate -> {new_lr}')
            return new_lr


class ClassificationEvent(DummyEvent):

    def __init__(self, stopper=10):
        self.stopper = stopper

    def __call__(self, **kwargs):
        # Earlystopping control
        if self.stopper:
            if kwargs['global_epoch'] == 1:
                kwargs['stopper'].freeze()
                kwargs['stopper'].reset()
                print(f"Epoch\t{kwargs['global_epoch']}: Earlystopping is frozen.")

            if kwargs['global_epoch'] < self.stopper:
                kwargs['stopper'].reset()

            if kwargs['global_epoch'] == self.stopper:
                kwargs['stopper'].unfreeze()
                print(f"Epoch\t{kwargs['global_epoch']}: Earlystopping is unfrozen.")
            
        # Loss function control
        if kwargs['criterion'].__class__.__name__ in [
                'OUSMLoss', 'JointOptimizationLoss', 'LabelSwappingOUSMLoss']:
            kwargs['criterion'].update(kwargs['global_epoch'])
        
        # Model update
        if isinstance(kwargs['model'], nn.DataParallel):
            model = kwargs['model'].module
        else:
            model = kwargs['model']
        if model.__class__.__name__ == 'IterativeSelfLearningModel':
            if kwargs['global_epoch'] > 1:
                model.correct_labels()
                print(model.train_pseudo_labels)
                print(model.valid_pseudo_labels)

    def __repr__(self):
        return f'Unfreeze(stopper={self.stopper})'


def analyse_results(predictions, labels, institutions=None, plot=False):
    qwk = QWK(6)
    overall_qwk = qwk(labels, predictions)
    if institutions is not None:
        karo_mask = institutions == 'karolinska'
        rad_mask = institutions == 'radboud'
        karo_qwk = qwk(labels[karo_mask], predictions[karo_mask])
        rad_qwk = qwk(labels[rad_mask], predictions[rad_mask])

        print(
            f'Overall QWK: {overall_qwk} / Karolinska: {karo_qwk} / Radboud: {rad_qwk}')
        print(f'\nOverall n={len(labels)}\n',
              confusion_matrix(labels, predictions))
        print(f'\nKarolinska n={karo_mask.sum()}\n', confusion_matrix(
            labels[karo_mask], predictions[karo_mask]))
        print(f'\nRadboud n={rad_mask.sum()}\n', confusion_matrix(
            labels[rad_mask], predictions[rad_mask]))

        if plot:
            sns.heatmap(confusion_matrix(labels, predictions,
                                         normalize='true'), annot=True, fmt='.3f')
            plt.title('Overall')
            plt.show()
            sns.heatmap(confusion_matrix(
                labels[karo_mask], predictions[karo_mask], normalize='true'), annot=True, fmt='.3f')
            plt.title('Karolinska')
            plt.show()
            sns.heatmap(confusion_matrix(
                labels[rad_mask], predictions[rad_mask], normalize='true'), annot=True, fmt='.3f')
            plt.title('Radboud')
            plt.show()
    else:
        print(f'Overall QWK: {overall_qwk}')
        print(f'\nOverall n={len(labels)}\n',
              confusion_matrix(labels, predictions))
        if plot:
            sns.heatmap(confusion_matrix(labels, predictions,
                                         normalize='true'), annot=True, fmt='.3f')
            plt.show()
