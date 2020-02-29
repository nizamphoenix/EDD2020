
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
def load_image(path):
    # returns an image of dtype int in range [0, 255]
    return np.asarray(Image.open(path))

def load_set(folder, shuffle=False):
    img_list = sorted(glob.glob(os.path.join(folder, '*.png')) + \
                      glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.tif')) + \
                      glob.glob(os.path.join(folder, '*.jpeg')))
    if shuffle:
        np.random.shuffle(img_list)
    data = []
    for img_fn in img_list:
        img = load_image(img_fn)
        data.append(img)
    return data, img_list
