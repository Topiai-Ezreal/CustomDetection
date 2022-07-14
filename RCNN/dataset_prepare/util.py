import os
import numpy as np
import xmltodict
import torch
import matplotlib.pyplot as plt
from xml.dom.minidom import parse


class_map = {1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat', 9:'chair',
             10:'cow',11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'pottedplant',
             17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor'}
class_map_opposite = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4, 'bottle':5, 'bus':6, 'car':7, 'cat':8,
                      'chair':9, 'cow':10, 'diningtable':11, 'dog':12, 'horse':13, 'motorbike':14, 'person':15,
                      'pottedplant':16, 'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
acc_dict = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0,
            11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, }


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_xml(xml_path):
    """
    解析xml文件，返回标注边界框坐标
    """
    bndboxes = []
    domTree = parse(os.path.join(xml_path))
    rootNode = domTree.documentElement                  # 文档根目录：annotation
    objects = rootNode.getElementsByTagName("object")   # 所有目标

    for a_object in objects:
        # 目标标签
        object_name = a_object.getElementsByTagName('name')[0].childNodes[0].data
        label = class_map_opposite[object_name]
        # 识别难易程度，0代表易，1代表难，不使用难识别的物体
        difficult = a_object.getElementsByTagName('difficult')[0].childNodes[0].data

        if difficult != '1':
            x_min = int(a_object.getElementsByTagName('xmin')[0].childNodes[0].data)
            y_min = int(a_object.getElementsByTagName('ymin')[0].childNodes[0].data)
            x_max = int(a_object.getElementsByTagName('xmax')[0].childNodes[0].data)
            y_max = int(a_object.getElementsByTagName('ymax')[0].childNodes[0].data)
            bndboxes.append((x_min, y_min, x_max, y_max, label))

    return np.array(bndboxes)


def parse_xml_svm(xml_path, target):
    """
    解析xml文件，返回标注边界框坐标
    """
    bndboxes = []
    domTree = parse(os.path.join(xml_path))
    rootNode = domTree.documentElement                  # 文档根目录：annotation
    objects = rootNode.getElementsByTagName("object")   # 所有目标

    for a_object in objects:
        # 目标标签
        object_name = a_object.getElementsByTagName('name')[0].childNodes[0].data
        if object_name == target:
            label = class_map_opposite[object_name]
            # 识别难易程度，0代表易，1代表难，不使用难识别的物体
            difficult = a_object.getElementsByTagName('difficult')[0].childNodes[0].data

            if difficult != '1':
                x_min = int(a_object.getElementsByTagName('xmin')[0].childNodes[0].data)
                y_min = int(a_object.getElementsByTagName('ymin')[0].childNodes[0].data)
                x_max = int(a_object.getElementsByTagName('xmax')[0].childNodes[0].data)
                y_max = int(a_object.getElementsByTagName('ymax')[0].childNodes[0].data)
                bndboxes.append((x_min, y_min, x_max, y_max, label))

    return np.array(bndboxes)


def iou(pred_box, target_box):
    """
    计算候选建议和标注边界框的IoU
    :param pred_box: 大小为[4]
    :param target_box: 大小为[N, 5]
    :return: [N]
    """
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    bnd_labels = target_box[:, -1]
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores, bnd_labels


def compute_ious(rects, bndboxs):
    iou_list = []
    for rect in rects:
        scores, labels = iou(rect, bndboxs)
        max_score, label = 0, 0
        for i in range(len(scores)):
            if scores[i] > max_score:
                max_score = scores[i]
                label = labels[i]
        iou_list.append((max_score, label))

    return iou_list


def save_model(model, model_save_path):
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title('loss')
    plt.savefig('./loss.png')
