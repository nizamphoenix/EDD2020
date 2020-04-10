from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import numpy as np
import glob
from PIL import Image

def load_image(path,is_mask):
    if not is_mask:
        return np.asarray(Image.open(path).convert("RGB"))
    else:
        return np.asarray(Image.open(path).convert('L'))

def load_set(folder,is_mask,shuffle=False):
    data = []
    img_list = sorted(glob.glob(os.path.join(folder, '*.png')) + \
                      glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.tif')) + \
                      glob.glob(os.path.join(folder, '*.jpeg')))
    if shuffle:
        np.random.shuffle(img_list)
    for img_fn in img_list:
        img = load_image(img_fn,is_mask)
        data.append(img)
    return data, img_list

class EDD(Dataset):
    '''
    Class for preparing the EDD2020 dataset
    '''
    def __init__(self, root, img_transform=None):
        self.root = root
        self.img_transform = img_transform
        self.original_images = None
        self.masks = None
        self.labels = None
        self._extract_images_and_segments(root)

    def __getitem__(self,index):
        img = self.original_images[index]
        mask = self.masks[index]
#         label = self.labels[index]
#         label = torch.as_tensor(label, dtype=torch.int32)
    
        if self.img_transform:
            img = self.img_transform(img)
            
        else:
            transform_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)
                #Normalizing makes images appear weird but necessary for resnet
            ])
            img = transform_to_tensor(img)
        
        return img, mask

    def __len__(self):
        return len(self.original_images)
    
    def _extract_images_and_segments(self,global_path):
        '''
        Function to process images and their respective masks.
        It sets  self.original_images and self.masks to processed images at the end.
        '''
        images_path = os.path.join(global_path, 'resized_images')
        all_images, img_filenames = load_set(folder=images_path,is_mask=False)
        img_filenames_with_ext = [os.path.split(fn)[-1] for fn in img_filenames]
        img_filenames_wo_ext = [fn[:fn.rfind('.')] for fn in img_filenames_with_ext]

        classes = ['BE','suspicious','HGD','cancer','polyp']

        masks_path = os.path.join(global_path, 'resized_masks')
        all_masks, mask_filenames = load_set(folder=masks_path,is_mask=True)
        mask_filenames_with_ext = [os.path.split(fn)[-1] for fn in mask_filenames]
        mask_filenames_wo_ext = [fn[:fn.rfind('.')] for fn in mask_filenames_with_ext]
        temp_dict={}#contains 502 mask filenames as keys and respective masks as values
        for i in range(len(all_masks)):
            temp_dict[mask_filenames_wo_ext[i]]=all_masks[i]

        all_masks=[]
        all_labels=[]
        for img in img_filenames_wo_ext:
            masks_for_img = []
            temp_labels = []
            for c in classes:
                try:
                    mask_file_name = img+'_'+c
                    temp_dict[mask_file_name] = np.where(temp_dict[mask_file_name] > 0, 1, 0)
                    temp_dict[mask_file_name] = temp_dict[mask_file_name].astype(np.float32)
                    masks_for_img.append(temp_dict[mask_file_name].reshape(temp_dict[mask_file_name].shape+ (1,)))
                    temp_labels.append(1)
                except KeyError:
                    dummy = np.zeros((224, 224)).astype(np.float32)
                    masks_for_img.append(dummy.reshape(dummy.shape + (1,)))
                    temp_labels.append(0)
            temp = None
            temp = np.concatenate(masks_for_img,2)#temp.shape (224, 224, 5)
            temp = temp.reshape((1,)+temp.shape)#temp.shape (1, 224, 224, 5)
            all_masks.append(temp)
            all_labels.append(temp_labels)
            
        all_masks = np.vstack(all_masks)#all_masks.shape (386, 224, 224, 5)
        all_masks = np.moveaxis(all_masks,source=3,destination=1)#all_masks.shape (386, 5, 224, 224)
        
        all_images = np.asarray(all_images)
        all_images = all_images.astype(np.uint8)
        augmented_images = []
        augmented_masks = []
        print(">>>>>>>>>>>>>>>>Augmenting to increase data size<<<<<<<<<<<<<<<<<<<<<<<<<")
        for image,masks in tqdm(zip(all_images,all_masks)):
            image = np.asarray(image)
            #print(image.shape,masks.shape)
            hflip_img = flip_img_horizontal(image)
            vflip_img = flip_img_vertical(image)
            temp_hor_masks = []#to store horizontally flipped masks for an image,which are later np.concatenated
            temp_ver_masks = []#to store vertically flipped masks for an image,which are later np.concatenated
            for mask in masks:
                temp_h = flip_mask_horizontal(mask).reshape((1,)+mask.shape)#reshaping (224,224)-->(1,224,224)
                temp_hor_masks.append(temp_h)
                temp_v = flip_mask_vertical(mask).reshape((1,)+mask.shape)#reshaping (224,224)-->(1,224,224)
                temp_ver_masks.append(temp_v)
            hflip_masks = np.concatenate(temp_hor_masks,0)#concatenating all 5 masks into 1 (224,224,5)
            vflip_masks = np.concatenate(temp_ver_masks,0)#concatenating all 5 masks into 1 (224,224,5)
            #print("flipped:",hflip_img.shape,hflip_masks.shape,vflip_img.shape,vflip_masks.shape)
            augmented_images.append(hflip_img)
            augmented_images.append(vflip_img)
            augmented_masks.append(hflip_masks.reshape((1,)+hflip_masks.shape))
            augmented_masks.append(vflip_masks.reshape((1,)+vflip_masks.shape))
            
        #images
        augmented_images = np.asarray(augmented_images)
        augmented_images = augmented_images.astype(np.uint8)
        print('orignial_images.shape',all_images.shape,'augmented_images.shape',augmented_images.shape)
        all_images = np.concatenate((all_images,augmented_images),axis=0)
        print('After augmentation, all_images.shape',all_images.shape)
        print("."*100)
        #masks
        augmented_masks = np.vstack(augmented_masks)
        print('orignial_masks.shape',all_masks.shape,'augmented_masks.shape',augmented_masks.shape)
        all_masks = np.concatenate((all_masks,augmented_masks),axis=0)
        print('After augmentation, all_masks.shape',all_masks.shape)
        print("*"*150)
        ################################results display
        print('len(all_images):',len(all_images),'len(all_masks):',len(all_masks))
        print('>>>>>>>>>>>Images<<<<<<<<<<<')
        print('type(all_images):',type(all_images),' all_images.shape:',all_images.shape)
        print('type(all_images[1]):',type(all_images[1]),' all_images[1].shape:',all_images[1].shape)
        print('.'*100)
        print('>>>>>>>>>>>Masks<<<<<<<<<<<<')
        print('type(all_masks):',type(all_masks),'all_masks.shape:',all_masks.shape)
        print('type(all_masks[1]):',type(all_masks[1]),'all_masks[1].shape:',all_masks[1].shape)
        print('.'*100)
        
        self.masks = all_masks
        self.original_images = all_images
        self.labels = all_labels
