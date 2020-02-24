import glob
global_path='./EDD2020/EDD2020_release-I_2020-01-15'
mask_filenames = sorted(glob.glob(os.path.join(global_path,'masks','*.tif')))
mask_images = []#target,response variable
for img_fn in mask_filenames:
  img = load_image(img_fn)
  mask_images.append(img)
for i in range(len(mask_images)):
  mask_images[i] = resize_image_to_square(mask_images[i], 224, pad_cval=0)
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
for i in range(len(X_all)):
  temp = []
  side = result_resolution[0]#224
  X_all[i] = resize_image_to_square(X_all[i], side, pad_cval=0)
  for c in range(len(classes)):
    try:
      index = mask_file_names_wo_ext.index(rel_file_names_wo_ext[i]+'_'+classes[c])
      temp.extend(mask_images_converted[index])
    except ValueError as er:
      temp.extend(np.zeros((224, 224)).reshape(np.zeros((224, 224)).shape + (1,)))
  target_masks.append(temp)
  print('--------------------------------------------------------------------------------------------')
print(np.array(target_masks).shape)

