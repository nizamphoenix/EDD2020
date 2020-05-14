from collections import defaultdict
import copy
import time
import torch
from meditorch.utils.plot import metrics_line, normalise_mask
import numpy as np


'''
modifying base trainer from meditorch library,
TODO: implement GroupKFold training
'''
def cal_dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice_coeff = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    dice_loss = 1 - dice_coeff
    return dice_loss.mean()

def intersection_over_union(target, pred):
    target = target.flatten()
    pred = pred.flatten()
    intersect = np.dot(target, pred)
    union = (target + pred).sum() - intersect
    return intersect/union

def calc_loss(pred, target, metrics, bce_weight):
    '''
    calculates total loss & other metrics like iou
    Total loss = bceloss.bce_weight + diceloss.(1-bce_weight)
    '''
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice_loss = cal_dice_loss(pred, target)
    total_loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)
    
    pred_binary = normalise_mask(pred.detach().cpu().numpy())
    iou = intersection_over_union(target.detach().cpu().numpy(), pred_binary)
    
    metrics['bce_loss']   += bce_loss.data.cpu().numpy() * target.size(0)
    metrics['dice_loss']  += dice_loss.data.cpu().numpy() * target.size(0)
    metrics['iou']        += iou * target.size(0)
    metrics['total_loss'] += total_loss.data.cpu().numpy() * target.size(0)
    
    return total_loss


def compute_metrics(metrics, epoch_samples):
    computed_metrics = {}
    for k in metrics.keys():
        computed_metrics[k] = metrics[k] / epoch_samples
    return computed_metrics

def print_metrics(computed_metrics, phase):
    outputs = []
    for k in computed_metrics.keys():
        outputs.append("{}:{:4f}".format(k, computed_metrics[k]))
    print("\t{}  -> {}".format(phase.ljust(5), " | ".join(outputs)))

class Trainer(object):

    def __init__(self, model, optimizer=None, scheduler=None):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Customized trainer\t",'Device====>',self.device)
        self.model = model.to(self.device)

        self.optimizer = optimizer
        if self.optimizer == None:
            print("optimizer undefined, hence using default Adam optimizer..")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.scheduler = scheduler
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)


    def train_model(self, dataloaders,bce_weight, num_epochs=20):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10
        epochs_metrics = {
            'train': [],
            'val': []
        }

        for epoch in range(num_epochs):
            print('Epoch {}/{}:'.format(epoch+1, num_epochs))

            since = time.time()

            # Each epoch has a training phase, followed by a validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("\tlearning rate: {:.2e}".format(param_group['lr']))
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                    
                metrics = defaultdict(float)
                epoch_samples = 0
                for inputs, targets in dataloaders[phase]:#phase is either train or val
                    masks = targets
                    #print('>>>>>>>>>>>',inputs.shape,masks.shape,'<<<<<<<<<<<<<<<<<<')
                    inputs = inputs.to(self.device)
                    masks = masks.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = calc_loss(outputs, masks, metrics,bce_weight)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    epoch_samples += inputs.size(0)

                computed_metrics = compute_metrics(metrics, epoch_samples)
                print_metrics(computed_metrics, phase)
                epochs_metrics[phase].append(computed_metrics)
                epoch_loss = metrics['total_loss'] / epoch_samples

                if phase == 'train':
                    self.scheduler.step()

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("\tCurrent epoch loss {:4f} is less than previous epoch loss {:4f}".format(epoch_loss, best_loss))
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            time_elapsed = time.time() - since
            print('\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('-' * 10)

        print('Best val loss: {:4f}'.format(best_loss))
        print('Saving best model')
        self.model.load_state_dict(best_model_wts)# load best model weights

        metrics_line(epochs_metrics)
