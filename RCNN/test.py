import numpy as np
import cv2

# negative_annotation_path = 'E:\\1_database\VOC\demo_finetune\Annotations\\2007_000528_1.txt'
# negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.int, delimiter=' ')
# print(len(negative_annotations))
# xmin, ymin, xmax, ymax, label = negative_annotations[0]
# print(xmin, ymin, xmax, ymax, label)

# class_map = {1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat', 9:'chair',
#              10:'cow',11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'pottedplant',
#              17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor'}
#
# tmp = ''
# for k in class_map.keys():
#     val = class_map[k]
#     tmp += "'" + val + "':" + str(k) + ', '
# print(tmp)

# t = np.loadtxt('E:\\1_database\VOC\VOC2012\VOC2012\ImageSets\Main\\aeroplane_train.txt', dtype=np.int, delimiter=' ')
# print(t)

image = cv2.imread('E:\\1_database\VOC\VOC2012\VOC2012\JPEGImages\\2008_003033.jpg', cv2.IMREAD_COLOR)
print(image.shape)