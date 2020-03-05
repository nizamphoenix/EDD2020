from torch.utils.data import DataLoader
import EDD
from torch.utils.data.sampler import SubsetRandomSampler
from meditorch.nn.models import UNetResNet
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from meditorch.nn import Trainer
from meditorch.utils.plot import plot_image_truemask_predictedmask
import numpy as np

def get_edd_loader(path,validation_split=.25,shuffle_dataset=True):
    '''
    v2 loader
    '''
    dataset = EDD(path)#instantiating the data set.
    dataset_size = 100#len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    loader={
        'train':DataLoader(dataset, batch_size=1, sampler=train_sampler),
        'val':DataLoader(dataset, batch_size= 5,sampler=valid_sampler)
    }
    return loader


def get_edd_loader(path,validation_split=.25,shuffle_dataset=True):
    '''
    v3
    ref:https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb
    '''
    dataset = EDD(path)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,collate_fn=utils.collate_fn)
    return data_loader,data_loader_test


#MAKING SENSE OF DATA LOADER
# for a_batch in data_loader:
#   #a_batch[0]: list of all images in the batch
#   #a_batch[1]: list of all targets in the batch
#   print(len(a_batch),len(a_batch[0]),len(a_batch[1]))
#   for image,target in zip(a_batch[0],a_batch[1]):
#     print(image.shape,target['masks'].shape)
#   print('<>'*200)
def main():
  np.random.seed(42)
  #seting up the data set
  loader = get_edd_loader('./EDD2020/EDD2020_release-I_2020-01-15',validation_split=.25,shuffle_dataset=True)
  
  #using UNet+ResNet combo
  model = UNetResNet(in_channel=3, n_classes=5)
  optimizer_func = optim.Adam(model.parameters(), lr=1e-4)
  scheduler = lr_scheduler.StepLR(optimizer_func, step_size=10, gamma=0.1)
  trainer = Trainer(model, optimizer=optimizer_func, scheduler=scheduler)
  #training
  trainer.train_model(loader, num_epochs=30)
  
  images, masks = next(iter(loader['val']))
  #predicting for only a batch of 5 from val set
  preds = trainer.predict(images)
  plot_image_truemask_predictedmask(images, masks, preds)


if __name__ == '__main__':
    main()
