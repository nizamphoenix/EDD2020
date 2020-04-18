# EDD2020  
Endoscopy is a widely used clinical procedure for the early etection of cancers in hollow-organs such as oesophagus, stomach, colon and bladder. Computer-assisted methods for accurate identification of tissues that are malignant, or potentially malignant has been at forefront of healthcare research to improve clinical diagnosis. Many methods to detect diseased regions in endoscopy have been proposed, however, these have primarily focussed on the task of polyp -- deleterious intestinal protrusion in the gastrointestinal tract -- detection on datasets that lack richness.  
The EDD2020 proposes a rich consolidated dataset from various leading research institues on cancer, under the supervision of cancer experts in collaboration with University of Oxford. This dataset not only focusses on just polyp detection but also on several other malignant tissues, leading to multi-class disease detection and segmentation challenge in clinical endoscopy. This challenge aims to establish a comprehensive dataset to benchmark algorithms for disease detection in endoscopy.  
Please refer https://edd2020.grand-challenge.org/  for details.  

This is a semantic segmentation task where an image can belong to either of the 5 labels: BE, suspicious, HGD, polyp, cancer.
The data preparation stage involves resizing the images to 224 x 224 size using OpenCV library, to leverage a myriad of computer vision algorithms for Machine learning.  
In addition, data augmentation approaches like flipping images horizontally, vertically; utilities in the Albumenations library are leveraged to increase the data size since the original data size is only 386.  
The current approach towards addressing the problem involves utilizing unet+resnet to identify segmentations(affected regions), and hence bounding-boxes(co-ordinates) thereafter.  
This approach is currenly giving a score = 0.1312, the more the better.  
Further improvements include using self-supervised learning to improve training.
