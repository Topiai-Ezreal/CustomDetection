import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset


class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        img_path = os.path.join(root_dir, 'JPEGImages')
        ann_path = os.path.join(root_dir, 'Annotations')
        name_list = os.listdir(img_path)
        positive_list = list()
        negative_list = list()

        for idx in range(len(name_list)):
            img_name = name_list[idx]

            # 记录正样本信息
            positive_annotation_path = os.path.join(ann_path, img_name[:-4] + '_1.txt')
            positive_annotations = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ')
            if len(positive_annotations.shape) == 1:    # 只有一个标注框
                if positive_annotations.shape[0] == 4:
                    positive_dict = dict()
                    positive_dict['rect'] = positive_annotations
                    positive_dict['image_id'] = idx
                    positive_list.append(positive_dict)
            else:                                       # 有多个标注框
                for positive_annotation in positive_annotations:
                    positive_dict = dict()
                    positive_dict['rect'] = positive_annotation
                    positive_dict['image_id'] = idx
                    positive_list.append(positive_dict)

            # 记录负样本信息
            negative_annotation_path = os.path.join(ann_path, img_name[:-4] + '_0.txt')
            negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.int, delimiter=' ')
            if len(negative_annotations.shape) == 1:  # 只有一个标注框
                if negative_annotations.shape[0] == 4:
                    negative_dict = dict()
                    negative_dict['rect'] = negative_annotations
                    negative_dict['image_id'] = idx
                    negative_list.append(negative_dict)
            else:  # 有多个标注框
                for negative_annotation in negative_annotations:
                    negative_dict = dict()
                    negative_dict['rect'] = negative_annotation
                    negative_dict['image_id'] = idx
                    negative_list.append(negative_dict)

        self.transform = transform
        self.name_list = name_list
        self.img_path = img_path
        self.positive_list = positive_list
        self.negative_list = negative_list

    def __getitem__(self, index: int):
        # 该index属于哪个图片，先寻找positive，再寻找negative
        if index < len(self.positive_list):
            positive_dict = self.positive_list[index]

            xmin, ymin, xmax, ymax, target = positive_dict['rect']
            img_id = positive_dict['image_id']
            img_name = self.name_list[img_id]
            image = cv2.imread(os.path.join(self.img_path, img_name))[ymin:ymax, xmin:xmax]
        else:
            negative_dict = self.negative_list[index - len(self.positive_list)]

            xmin, ymin, xmax, ymax, target = negative_dict['rect']
            img_id = negative_dict['image_id']
            img_name = self.name_list[img_id]
            image = cv2.imread(os.path.join(self.img_path, img_name))[ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image)

        return image, int(target)

    def __len__(self) -> int:
        return len(self.positive_list) + len(self.negative_list)

    def get_positive_num(self) -> int:
        return len(self.positive_list)

    def get_negative_num(self) -> int:
        return len(self.negative_list)

    def get_img_num(self) -> int:
        return len(self.name_list)


def tes(idx):
    root_path = 'E:\\1_database\\VOC\\demo_finetune'
    train_data_set = CustomFinetuneDataset(root_path)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    # 测试id=3/66516/66517/530856
    image, target = train_data_set.__getitem__(idx)
    print('target: %d' % target)

    image = Image.fromarray(image)
    image.show()
