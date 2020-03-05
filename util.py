
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
