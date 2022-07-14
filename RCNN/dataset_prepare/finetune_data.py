import time
import shutil
import numpy as np
import cv2
import os
from selectivesearch import selective_search
from util import check_dir, parse_xml, compute_ious


def parse_annotation_jpeg(annotation, image):
    """
    获取正负样本（注：忽略difficult的标注框）
    正样本：proposal与gt box的IoU大于等于0.5
    负样本：剩余的其它proposal
    """
    rects = selective_search(image, strategy='quality')     # 提取proposals
    gtboxs = parse_xml(annotation)                          # 获取ground truth

    # 计算proposal和gt的IoU
    iou_list = compute_ious(rects, gtboxs)

    positive_list = []
    negative_list = []
    for i in range(len(iou_list)):
        if iou_list[i][0] >= 0.5:
            rect = list(rects[i])
            rect.append(iou_list[i][1])
            positive_list.append(rect)
        else:
            if len(negative_list) < 2000:
                rect = list(rects[i])
                rect.append(0)
                negative_list.append(rect)
            else:
                continue

    return positive_list, negative_list


if __name__ == '__main__':
    root_path = 'E:\\1_database\\VOC\\RCNN\\'
    ann_path = os.path.join(root_path, 'Annotations')
    img_path = os.path.join(root_path, 'JPEGImages')
    ImageSets_path = os.path.join(root_path, 'ImageSets', 'Main')  # 官方划分的训练/验证, txt文件
    proposal_path = os.path.join(root_path, 'Proposals_cnn')
    check_dir(proposal_path)

    since = time.time()
    for dataset in ['train', 'val']:
        dataset_path = os.path.join(proposal_path, dataset)
        check_dir(dataset_path)
        img_num = 0
        total_num_positive, total_num_negative = 0, 0

        main_set = os.path.join(ImageSets_path, dataset + '.txt')
        with open(main_set) as w:
            for line in w.readlines():
                name = line[:-1]
                image = cv2.imread(os.path.join(img_path, name + '.jpg'), cv2.IMREAD_COLOR)
                if image.shape[0] < 210 or image.shape[1] < 210:
                    continue
                annotation = os.path.join(ann_path, name + '.xml')
                positive_list, negative_list = parse_annotation_jpeg(annotation, image)
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
