
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

def resize_images(src,dest,image_size):
    '''
    credits: https://evigio.com/post/resizing-images-into-squares-with-opencv-and-python
    '''
    import cv2
    import os
    import numpy as np

    i = 1
    img_size = image_size

    path = src

    for img_name in sorted(os.listdir(path)):
        try:
            img = cv2.imread(os.path.join(path, img_name))

            h, w = img.shape[:2]
            a1 = w/h
            a2 = h/w

            if(a1 > a2):
                print('a1 > a2')
                # if width greater than height
                r_img = cv2.resize(img, (round(img_size * a1), img_size), interpolation = cv2.INTER_AREA)
                margin = int(r_img.shape[1]/6)
                crop_img = r_img[0:img_size, margin:(margin+img_size)]

            elif(a1 < a2):
                print('a1 < a2')
                # if height greater than width
                r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
                margin = int(r_img.shape[0]/6)
                crop_img = r_img[margin:(margin+img_size), 0:img_size]

            elif(a1 == a2):
                print('a1== a2')
                # if height and width are equal
                r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
                crop_img = r_img[0:img_size, 0:img_size]

            if(crop_img.shape[0] != img_size or crop_img.shape[1] != img_size):
                print('someting....')
                crop_img = r_img[0:img_size, 0:img_size]

            if(crop_img.shape[0] == img_size and crop_img.shape[1] == img_size):

                print("Saving image with dims: " + str(crop_img.shape[0]) + "x" + str(crop_img.shape[1]))
                cv2.imwrite(dest+img_name, crop_img)
                i += 1
            print('><'*20)
        except:
            print('Could not save image.')
