from torch.utils.data import Dataset
from torchvision import transforms
from util import load_set
import torch
import os
import numpy as np
import glob

class EDD(Dataset):
    '''
    Class for preparing the EDD2020 dataset
    '''
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.original_images = None
        self.masks = None
        self.labels = None
        self._extract_images_and_segments(root)

    def __getitem__(self,index):
        img = self.original_images[index]
        mask = self.masks[index]
        label = self.labels[index]
        boxes = []
        for i in range(len(mask)):
            mask[i] = np.asarray(mask[i])
            pos  = np.where(mask[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        image_id = torch.tensor([index])
        
        if self.transform:
            img = self.transform(img)
        else:
            transform_to_tensor = transforms.Compose([transforms.ToTensor(),])
            img = transform_to_tensor(img)
            
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = label
        target["masks"] = mask
        target["image_id"] = image_id

        return img, target

    def __len__(self):
        return len(self.original_images)
    
    def _extract_images_and_segments(self,global_path):
        '''
        Function to process images and their respective masks.
        It sets  self.original_images and self.masks to processed images at the end.
        '''
        images_path = os.path.join(global_path, 'originalImages')
        all_images, img_filenames = load_set(folder=images_path,is_mask=False)
        img_filenames2 = [os.path.split(fn)[-1] for fn in img_filenames]
        img_filenames_wo_ext = [fn[:fn.rfind('.')] for fn in img_filenames2]

        classes = ['BE','suspicious','HGD','cancer','polyp']

        masks_path = os.path.join(global_path, 'masks')
        all_masks, mask_filenames = load_set(folder=masks_path,is_mask=True)
        mask_filenames2 = [os.path.split(fn)[-1] for fn in mask_filenames]
        mask_filenames_wo_ext = [fn[:fn.rfind('.')] for fn in mask_filenames2]
        temp_dict={}#contains 502 mask filenames as keys and respective masks as values
        for i in range(len(all_masks)):
            temp_dict[mask_filenames_wo_ext[i]]=all_masks[i]

        all_masks=[]
        all_labels=[]
        for img in img_filenames_wo_ext:
            temp_masks = []
            temp_labels = []
            for c in classes:
                try:
                    temp_masks.append(temp_dict[img+'_'+c])
                    temp_labels.append(1)
                except KeyError:
                    temp_labels.append(0)
                    continue
            all_masks.append(temp_masks)#appending the images directly
            #all_masks.append(np.array(temp_masks))
            all_labels.append(temp_labels)
#         print('len(all_images):',len(all_images),'len(all_masks):',len(all_masks),' len(all_labels):',len(all_labels))
#         print('type(all_images[1]):',type(all_images[1]),' all_images[1].shape:',all_images[1].shape)
#         print('type(all_masks[1]):',type(all_masks[1]),'all_masks[1].shape:',all_masks[1].shape)
#         print('all_labels[1]: ',all_labels[1])
        self.masks = all_masks
        self.original_images = all_images
        self.labels = all_labels
