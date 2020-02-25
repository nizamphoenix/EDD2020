from torch.utils.data import DataLoader
import EDD
from torch.utils.data.sampler import SubsetRandomSampler
from meditorch.nn.models import UNetResNet
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from meditorch.nn import Trainer
from meditorch.utils.plot import plot_image_truemask_predictedmask

def main():
  #seting up the data set
  dataset = EDD()
  #splitting the data set into train,val
  batch_size = 5
  validation_split = .2
  shuffle_dataset = True
  random_seed= 42
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  loader={
      'train':DataLoader(dataset, batch_size=batch_size, sampler=train_sampler),
      'val':DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler)
  }
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
