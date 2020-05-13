from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from meditorch.nn.models import UNetResNet
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from meditorch.nn import Trainer
from meditorch.utils.plot import plot_image_truemask_predictedmask
import numpy as np
import EDD
from util import resize_images

np.random.seed(42)

def get_edd_loader(path,validation_split=.25,shuffle_dataset=True):
    dataset = EDD(path)#instantiating the data set.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    loader={
        'train':DataLoader(dataset, batch_size=4, sampler=train_sampler),
        'val':DataLoader(dataset, batch_size = 4,sampler=valid_sampler)
    }
    return loader


def main():
  np.random.seed(42)
  #seting up the data set
  !mkdir ./EDD2020/resized_masks/
  resize_my_images('./EDD2020/EDD2020_release-I_2020-01-15/masks/','./EDD2020/resized_masks/',is_masks=True)
  !mkdir ./EDD2020/resized_images/
  resize_my_images('./EDD2020/EDD2020_release-I_2020-01-15/originalImages/','./EDD2020/resized_images/',is_masks=False)
  
    
    
  loader = get_edd_loader('./EDD2020/',validation_split=.25,shuffle_dataset=True)
  
  #using UNet+ResNet combo
  model = UNetResNet(in_channel=3, n_classes=5)
  optimizer_func = optim.Adam(model.parameters(), lr=1e-4)
  scheduler = lr_scheduler.StepLR(optimizer_func, step_size=10, gamma=0.1)
  trainer = Trainer(model, optimizer=optimizer_func, scheduler=scheduler)
  #training
  trainer.train_model(loader, num_epochs=30)
  
  images, masks = next(iter(loader['val']))
  #predicting for only a batch of 4 from val set
  preds = trainer.predict(images)
  plot_image_truemask_predictedmask(images, masks, preds)


if __name__ == '__main__':
    main()
