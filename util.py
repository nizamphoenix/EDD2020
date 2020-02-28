
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:24:24 2018

@author: ead2019
"""
def convert_boxes(boxes, class_names, datatype, imgshape):
    nrows, ncols = imgshape
    data = []

    if len(boxes) > 0:
        for bbox in boxes:
            if datatype=='GT':
                cls, b1, b2, b3, b4 = bbox
            elif datatype=='Pred':
                cls, conf, b1, b2, b3, b4 = bbox
            else:
                raise Exception('datatype should be either \'GT\' or \'Pred\'. The value of datatype was: {}'.format(datatype))

            # check whether we have been given the name already.
            try:
                cls = int(cls)
                cls_name = class_names[int(cls)]
            except:
                cls_name = str(cls)

            # check whether yolo or not.
            bbox_bounds = np.hstack([b1,b2,b3,b4]).astype(np.float)

            if bbox_bounds.max() < 1.1:
                # yolo:
#                x1 = (bbox_bounds[0] - bbox_bounds[2]) / 2. * ncols
#                y1 = (bbox_bounds[1] - bbox_bounds[3]) / 2. * nrows
#                x2 = (bbox_bounds[0] + bbox_bounds[2]) / 2. * ncols
#                y2 = (bbox_bounds[1] + bbox_bounds[3]) / 2. * nrows
                                
                # yolo 2 voc
                x = bbox_bounds[0] * ncols
                w = bbox_bounds[2] * ncols
                
                y = bbox_bounds[1] * nrows
                h = bbox_bounds[3] * nrows
                
                x1 = x - w/2
                x2 = x + w/2
                y1 = y - h/2
                y2 = y + h/2
       
            else:
                # assume voc:
                x1,y1,x2,y2 = bbox_bounds

            # clip to image bounds
            x1 = int(np.clip(x1, 0, ncols-1))
            y1 = int(np.clip(y1, 0, nrows-1))
            x2 = int(np.clip(x2, 0, ncols-1))
            y2 = int(np.clip(y2, 0, nrows-1))

            # strictly speaking we should have the following but we can implement a filter instead.
            # assert(x2>x1 and y2>y1) # check this is true! for voc
            if x2>x1 and y2>y1:
                # only append if this constraint is satisfied.
                if datatype=='GT':
                    data.append([cls_name, x1, y1, x2, y2])
                elif datatype=='Pred':
                    data.append([cls_name, float(conf), x1,y1,x2,y2])

        if len(data) > 0:
            return np.vstack(data) # create an array.
        else:
            return data
    else:
        return data

def read_img(imfile):
    import cv2
    return cv2.imread(imfile)[:,:,::-1]


def read_boxes(txtfile):

    import numpy as np
    lines = []

    with open(txtfile, "r") as f:
        for line in f:
            #print('line:',line)
            line = line.strip()
            line_list = line.split()
            try:
                cls = line_list[4]
                coords = line_list[:4]
                val = None
                if cls=='BE':
                    val=0
                elif cls=='suspicious':
                    val=1
                elif cls=='HGD':
                    val=2
                elif cls=='cancer':
                    val=3
                elif cls=='polyp':
                    val=4
                newline=[]
                newline.append(str(val))
                newline.extend(coords)
                box = np.hstack(newline).astype(np.float)
                box[0] = int(box[0])
                #print("coordinates:",box)
                lines.append(box)
            except:
                print('ERRRRRRRRRRR')
    return np.array(lines)




def plt_rectangle(plt,label,x1,y1,x2,y2):
  '''  
  plt   : matplotlib.pyplot object
  label : string containing the object class name
  x1    : top left corner x coordinate
  y1    : top left corner y coordinate
  x2    : bottom right corner x coordinate
  y2    : bottom right corner y coordinate
  '''
  linewidth = 3
  color = "yellow"
  plt.text(x1,y1,label,fontsize=20,backgroundcolor="magenta")
  plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color)
  plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color)
  plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color)
  plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color)

def plot_boxes(ax, boxes, labels):
    for b in boxes:
        col = None
        cls, x1, y1, x2, y2 = b
        if cls==0:
            col='cyan'
        elif cls==1:
            col='blue'
        elif cls==2:
            col='green'
        elif cls==3:
            col='red'
        elif cls==4:
            col='black'
        print(x1,y1,x2,y2)
        y1=512-y1
        y2=512-y2
        print(x1,y1,x2,y2)
        x1=int(np.clip(x1, 0, 224-1))
        x2=int(np.clip(x2, 0, 224-1))
        y1=int(np.clip(y1, 0, 224-1))
        y2=int(np.clip(y2, 0, 224-1))
        print(x1,y1,x2,y2)
        ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1],lw=2, color=col)
    return []


def read_obj_names(textfile):
    
    import numpy as np 
    classnames = []
    
    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line)>0:
                classnames.append(line)
            
    return np.hstack(classnames)


if __name__=="__main__":
    
    """
    Example script to read and plot bounding box  (which are provided in <x1,y1,x2,y2> (VOC)format)
    """
    import pylab as plt
    import sys
    imgfile='./EDD2020/EDD2020_release-I_2020-01-15/originalImages/EDD2020_ACB0001.jpg'
    bboxfile='./EDD2020/EDD2020_release-I_2020-01-15/bbox/EDD2020_ACB0001.txt'
    masksfile1='./EDD2020/EDD2020_release-I_2020-01-15/masks/EDD2020_ACB0001_BE.tif'
    masksfile2='./EDD2020/EDD2020_release-I_2020-01-15/masks/EDD2020_ACB0001_suspicious.tif'
    classfile = './EDD2020/EDD2020_release-I_2020-01-15/class_list.txt'

    
    img = read_img(imgfile)
    img = resize_image_to_square(img, 224, pad_cval=0)
    boxes = read_boxes(bboxfile)
    mask1 = read_img(masksfile1)
    mask1 = resize_image_to_square(mask1, 224, pad_cval=0)
    mask2 = read_img(masksfile2)
    mask2 = resize_image_to_square(mask2, 224, pad_cval=0)
    classes = read_obj_names(classfile)
    #boxes=convert_boxes(boxes,['0','1','2','3','4'],'GT',img.shape[:2])
    #boxes=boxes.astype(np.float)
    fig, ax = plt.subplots(nrows=1,ncols=3)
    plot_boxes(ax[0], boxes, classes)
    ax[0].imshow(img)
    ax[1].imshow(mask1)
    ax[2].imshow(mask2)
    plt.show()
