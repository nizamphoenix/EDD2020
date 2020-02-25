import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from meditorch.utils.images import load_set, load_image, resize_image_to_square
import glob

class EDD(Dataset):
    '''
    Class for reading and loading the EDD dataset
    '''
    def __init__(self,transform=None,result_resolution=(224,224)):
      self.transform = transform
      self.original_images = None
      self.masks = None
      self._extract_images_and_segments(result_resolution)

    def __getitem__(self,index):
      '''
      This is a magic function & is called when an object of EDD is accessed as a list.
      It returns image and its mask
      Example:
      obj=EDD()
      obj[index]---calls--->obj.__getitem__(index)
      '''
      img, mask = self.original_images[index], self.masks[index]
      if self.transform:
        img = self.transform(img)
      else:
        transform_to_tensor = transforms.Compose([transforms.ToTensor(),])
        img = transform_to_tensor(img)
      return img, mask

    def __len__(self):
      return len(self.original_images)

    def _extract_images_and_segments(self,result_resolution):
      '''
      Function to process images and their respective masks.
      It sets  self.original_images and self.masks to processed images at the end.
      '''
      global_path='./EDD2020/EDD2020_release-I_2020-01-15'
      mask_filelist = sorted(glob.glob(os.path.join(global_path,'masks','*.tif')))
      mask_file_names = [os.path.split(fn)[-1] for fn in mask_filelist]
      mask_file_names_wo_ext = [fn[:fn.rfind('.')] for fn in mask_file_names]
      mask_images = []#target,response variable
      for img_fn in mask_filelist:
        img = load_image(img_fn)
        mask_images.append(img)
      for i in range(len(mask_images)):
        mask_images[i] = resize_image_to_square(mask_images[i], result_resolution[0], pad_cval=0)
        mask_images[i] = mask_images[i].reshape(mask_images[i].shape + (1,))
      mask_images_converted=[]
      mask_images_converted = np.asarray(mask_images)#converting the list to numpy array
      mask_images_converted = np.where(mask_images_converted > 0, 1, 0)
      mask_images_converted = mask_images_converted.astype(np.float32)

      images_path = os.path.join(global_path, 'originalImages')
      X_all, file_names = load_set(folder=images_path)
      rel_file_names = [os.path.split(fn)[-1] for fn in file_names]
      rel_file_names_wo_ext = [fn[:fn.rfind('.')] for fn in rel_file_names]
      classes = ['BE','suspicious','HGD','cancer','polyp']
      target_masks = []
      temp2=[]
      for i in range(len(X_all)):
        temp = []
        side = result_resolution[0]#224
        X_all[i] = resize_image_to_square(X_all[i], side, pad_cval=0)
        for c in range(len(classes)):
          try:
            index = mask_file_names_wo_ext.index(rel_file_names_wo_ext[i]+'_'+classes[c])
            #print(mask_images_converted[index].shape)
            temp.append(mask_images_converted[index])
          except ValueError as er:
            temp.append(np.zeros((224, 224)).reshape(np.zeros((224, 224)).shape + (1,)))
        temp2=np.concatenate(temp,2)
        temp2= temp2.reshape((1,)+temp2.shape)
        #print(np.array(temp2).shape)
        #print('--------------------------------------------------------------------------------------------')
        target_masks.append(temp2)
      target_masks=np.vstack(target_masks)
      target_masks=np.moveaxis(target_masks,source=3,destination=1)
      input_images = np.asarray(X_all)
      input_images = input_images.astype(np.uint8)
      print('input_images.shape',input_images.shape,'target_masks.shape',target_masks.shape)
      self.original_images = input_images
      self.masks = target_masks
      print('Completed processing original images and target masks.')
