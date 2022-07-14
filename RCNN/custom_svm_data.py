import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CustomSVMDataset(Dataset):
    def __init__(self, root_dir, label, dataset, transform=None):
        img_path = os.path.join(root_dir, 'JPEGImages')
        proposal_path = os.path.join(root_dir, 'Proposals', label, dataset)

        # 记录图片名称
        name_list = list()
        for file in os.listdir(proposal_path):
            file = str(file[:-6])
            if file not in name_list:
                name_list.append(file)

        positive_list = list()
        negative_list = list()

        for name in name_list:

            # 记录正样本信息
            positive_annotations = np.loadtxt(os.path.join(proposal_path, name + '_1.txt'), dtype=np.int, delimiter=' ')
            if len(positive_annotations.shape) == 1:    # 只有一个标注框
                if positive_annotations.shape[0] == 5:
                    positive_dict = dict()
                    positive_dict['rect'] = positive_annotations
                    positive_dict['image_id'] = name
                    positive_list.append(positive_dict)
            else:
                for positive_annotation in positive_annotations:
                    positive_dict = dict()
                    positive_dict['rect'] = positive_annotation
                    positive_dict['image_id'] = name
                    positive_list.append(positive_dict)

            # 记录负样本信息
            negative_annotations = np.loadtxt(os.path.join(proposal_path, name + '_0.txt'), dtype=np.int, delimiter=' ')
            if len(negative_annotations.shape) == 1:  # 只有一个标注框
                if negative_annotations.shape[0] == 5:
                    negative_dict = dict()
                    negative_dict['rect'] = negative_annotations
                    negative_dict['image_id'] = name
                    negative_list.append(negative_dict)
            else:  # 有多个标注框
                for negative_annotation in negative_annotations:
                    negative_dict = dict()
                    negative_dict['rect'] = negative_annotation
                    negative_dict['image_id'] = name
                    negative_list.append(negative_dict)
        print(positive_list)

        self.transform = transform
        self.img_path = img_path
        self.proposal_path = proposal_path
        self.name_list = name_list
        self.positive_list = positive_list
        self.negative_list = negative_list

    def __getitem__(self, index: int):
        # 该index属于哪个图片，先寻找positive，再寻找negative
        if index < len(self.positive_list):
            positive_dict = self.positive_list[index]

            xmin, ymin, xmax, ymax, target = positive_dict['rect']
            img_name = positive_dict['image_id'] + '.jpg'
            img = cv2.imread(os.path.join(self.img_path, img_name))[ymin:ymax, xmin:xmax]
            cache_dict = positive_dict
        else:
            negative_dict = self.negative_list[index - len(self.positive_list)]

            xmin, ymin, xmax, ymax, target = negative_dict['rect']
            img_name = negative_dict['image_id'] + '.jpg'
            img = cv2.imread(os.path.join(self.img_path, img_name))[ymin:ymax, xmin:xmax]
            cache_dict = negative_dict

        if self.transform:
            img = self.transform(img)

        return img, int(target), cache_dict

    def __len__(self) -> int:
        return len(self.positive_list) + len(self.negative_list)

    def get_transform(self):
        return self.transform

    def get_positive_num(self) -> int:
        return len(self.positive_list)

    def get_negative_num(self) -> int:
        return len(self.negative_list)

    def get_positives(self) -> list:
        return self.positive_list

    def get_negatives(self) -> list:
        return self.negative_list

    def get_name_list(self) -> list:
        return self.name_list

    # 用于hard negative mining
    # 替换负样本
    def set_negative_list(self, negative_list):
        self.negative_list = negative_list


def tes(idx):
    root_path = 'E:\\1_database\\VOC\\RCNN'
    train_data_set = CustomSVMDataset(root_path, label='aeroplane', dataset='train')

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    # 测试id=3/66516/66517/530856
    image, target, cache_dict = train_data_set.__getitem__(idx)
    print('target: %d' % target)
    print('dict: ' + str(cache_dict))

    image = Image.fromarray(image)
    image.show()
