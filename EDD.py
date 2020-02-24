import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from meditorch.utils.images import load_set, load_image, resize_image_to_square

class EDD(Dataset):
  '''
  Class for reading and loading the EDD dataset
  '''
    def __init__(self, root, train=True, transform=None,result_resolution=(224,224)):
        self.root = root
        self.transform = transform
        self.original_images = None
        self.masks = None
        self._extract_images_and_segments(train, result_resolution)

    def __getitem__(self, index):
        '''
        This is a magic function & is called when an object of EDD is accessed as a list.
        It returns image and its mask
        Example:
        obj=EDD()
        obj[index]---calls--->obj.__getitem__(index)
        '''
        img, mask = self.data[index], self.masks[index]
        if self.transform:
            img = self.transform(img)
        else:
            transform_to_tensor = transforms.Compose([transforms.ToTensor(),])
            img = transform_to_tensor(img)
        return img, mask

    def __len__(self):
        return len(self.data)

    def _extract_images_and_segments(self, is_train, return_disc, return_cup, result_resolution):
        global_path='./EDD2020/EDD2020_release-I_2020-01-15'
        BE, suspicious, HGD, cancer, polyp = [], [], [], [], []
        images_path = os.path.join(global_path, 'originalImages')
        X_all, file_names = load_set(folder=images_path)
        rel_file_names = [os.path.split(fn)[-1] for fn in file_names]
        rel_file_names_wo_ext = [fn[:fn.rfind('.')] for fn in rel_file_names]

        for fn in rel_file_names_wo_ext:
            mask_folder = 'masks'
            disc_segmn = load_image(os.path.join(global_path, mask_folder, fn,'_','.*','.tif'))
            disc_all.append(disc_segmn)

        
        for i in range(len(X_all)):
            side = result_resolution[0]#224
            X_all[i] = resize_image_to_square(X_all[i], side, pad_cval=0)

            if return_disc:
                disc_all[i] = resize_image_to_square(disc_all[i], side, pad_cval=0)
                disc_all[i] = disc_all[i].reshape(disc_all[i].shape + (1,))
            if return_cup:
                cup_all[i] = resize_image_to_square(cup_all[i], side, pad_cval=0)
                cup_all[i] = cup_all[i].reshape(cup_all[i].shape + (1,))

        input_images = np.asarray(X_all)
        input_images = input_images.astype(np.uint8)

        if return_disc:
            disc_converted = np.asarray(disc_all)
            disc_converted = np.where(disc_converted > 0, 1, 0)
            disc_converted = disc_converted.astype(np.float32)

        if return_cup:
            cup_converted = np.asarray(cup_all)
            cup_converted = np.where(cup_converted > 0, 1, 0)
            cup_converted = cup_converted.astype(np.float32)

        BE, suspicious, HGD, cancer, polyp
        target_masks = np.concatenate((disc_converted, cup_converted), axis=3)
       
        target_masks = np.rollaxis(target_masks, 3, 1)

        self.original_images = input_images
        self.masks = target_masks

        print('Completed extracting `data` and `targets`.')

    
