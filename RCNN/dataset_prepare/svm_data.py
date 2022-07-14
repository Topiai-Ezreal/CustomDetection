import time
import shutil
import numpy as np
import cv2
import os
from dataset_prepare.selectivesearch import selective_search
from util import check_dir, parse_xml_svm, compute_ious


# class_map = {1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat', 9:'chair',
#              10:'cow',11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'pottedplant',
#              17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor'}
class_map = {1:'aeroplane'}


def parse_annotation_jpeg(annotation, image, label):
    """
    获取正负样本（注：忽略difficult的标注框）
    正样本：gt box
    负样本：0<iou<=0.3的proposal
    """
    rects = selective_search(image, strategy='fast')     # 提取proposals
    gtboxs = parse_xml_svm(annotation, label)               # 获取ground truth

    # 计算proposal和gt的IoU
    iou_list = compute_ious(rects, gtboxs)

    positive_list = gtboxs
    negative_list = []
    for i in range(len(iou_list)):
        if 0 < iou_list[i][0] <= 0.3:
            rect = list(rects[i])
            rect.append(0)
            negative_list.append(rect)

    return positive_list, negative_list


if __name__ == '__main__':
    root_path = 'E:\\1_database\\VOC\\RCNN'
    ann_path = os.path.join(root_path, 'Annotations')               # VOC标注, xml文件
    img_path = os.path.join(root_path, 'JPEGImages')                # 所有图片
    ImageSets_path = os.path.join(root_path, 'ImageSets', 'Main')   # 官方划分的训练/验证, txt文件
    proposal_path = os.path.join(root_path, 'Proposals')            # 保存提取的proposal信息
    check_dir(proposal_path)

    for k in class_map.keys():
        label = class_map[k]
        class_path = os.path.join(proposal_path, label)
        check_dir(class_path)
        print('-' * 10, str(k) + ': ' + label, '-' * 10)

        since = time.time()
        for dataset in ['train', 'val']:
            dataset_path = os.path.join(class_path, dataset)
            check_dir(dataset_path)
            img_num = 0
            total_num_positive, total_num_negative = 0, 0

            main_set = os.path.join(ImageSets_path, label + '_' + dataset + '.txt')
            with open(main_set) as w:
                for line in w.readlines():
                    name = line.split(' ')[0]
                    exist = line.split(' ')[-1][:-1]
                    if exist == '1':
                        image = cv2.imread(os.path.join(img_path, name + '.jpg'), cv2.IMREAD_COLOR)
                        if image.shape[0] < 200:
                            continue
                            # image = cv2.resize(image, (image.shape[1], image.shape[1]))
                        annotation = os.path.join(ann_path, name + '.xml')
                        positive_list, negative_list = parse_annotation_jpeg(annotation, image, label)
                        total_num_positive += len(positive_list)
                        total_num_negative += len(negative_list)

                        positive_proposal_path = os.path.join(dataset_path, name + '_1.txt')
                        negative_proposal_path = os.path.join(dataset_path, name + '_0.txt')
                        # 保存正负样本标注
                        np.savetxt(positive_proposal_path, np.array(positive_list), fmt='%d', delimiter=' ')
                        np.savetxt(negative_proposal_path, np.array(negative_list), fmt='%d', delimiter=' ')

                        img_num += 1

            print(dataset, 'img_num:', img_num)
            print(dataset, 'positive num: %d' % total_num_positive)
            print(dataset, 'negative num: %d' % total_num_negative)

        time_elapsed = time.time() - since
        minute = str(int(time_elapsed // 60))
        sec = str(time_elapsed % 60)
        print(minute + 'min', sec)
