
import os
import numpy as np
from PIL import Image
from skimage.transform import resize

def load_image(path,is_mask):
    if not is_mask:
        return Image.open(path).convert("RGB")
    else:
        return Image.open(path)

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

    

def resize_my_images(src,dst,is_masks):
    '''
    ref: https://evigio.com/post/resizing-images-into-squares-with-opencv-and-python
    '''
    import cv2
    import os
    import numpy as np

    i = 0
    img_size = 224
    path = src
    for img_name in sorted(os.listdir(path)):
        img = None
#         print(img_name)
        try:
            if not is_masks:
                img = cv2.imread(os.path.join(path, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif is_masks:
                img = cv2.imread(os.path.join(path, img_name),cv2.IMREAD_GRAYSCALE)
    
            h, w = img.shape[:2]
            a1 = w/h
            a2 = h/w

            if(a1 > a2):
                #print('a1 > a2')
                # if width greater than height
                r_img = cv2.resize(img, (round(img_size * a1), img_size), interpolation = cv2.INTER_AREA)
                margin = int(r_img.shape[1]/6)
                crop_img = r_img[0:img_size, margin:(margin+img_size)]

            elif(a1 < a2):
                #print('a1 < a2')
                # if height greater than width
                r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
                margin = int(r_img.shape[0]/6)
                crop_img = r_img[margin:(margin+img_size), 0:img_size]

            elif(a1 == a2):
                #print('a1== a2')
                # if height and width are equal
                r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
                crop_img = r_img[0:img_size, 0:img_size]

            if(crop_img.shape[0] != img_size or crop_img.shape[1] != img_size):
                #print('someting....')
                crop_img = r_img[0:img_size, 0:img_size]

            if(crop_img.shape[0] == img_size and crop_img.shape[1] == img_size):
                if not is_masks:
                    cv2.imwrite(dst + img_name, crop_img)#SAVING AS RGB FORMAT 
                    #display_image(crop_img,is_mask=False)
                elif is_masks:
                    #converting masks: tif--->jpg   
                    cv2.imwrite(dst + img_name.split('.')[0]+'.jpg', crop_img)
                    #display_image(crop_img,is_mask=True)
                #print("Saving "+ img_name.split('.')[0] + " with dims: " + str(crop_img.shape))
                i += 1
            #print('><'*20)
        except:
            print('Could not save image.')
    if is_masks:
        print('no of masks saved: ',i)
    else:
        print('no of images saved: ',i)
        
        
def display_image(img,is_mask=False):
    '''
    To display image/mask 
    args: img, image
          is_mask, boolean True if greyscale mask is passed
    '''
    from matplotlib import pyplot as plt
    %matplotlib inline
    if is_mask:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()            
            
