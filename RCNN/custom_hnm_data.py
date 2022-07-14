import cv2
import os
import torch.nn as nn
from torch.utils.data import Dataset
from custom_svm_data import CustomSVMDataset


class CustomHNMDataset(Dataset):
    def __init__(self, negative_list, name_list, transform=None):
        self.negative_list = negative_list
        self.transform = transform

    def __getitem__(self, index: int):
        target = 0
        negative_dict = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dict['rect']
        img_id = negative_dict['image_id']
        img_name = name_list[img_id]
        img = cv2.imread(os.path.join(img_path, img_name))[ymin:ymax, xmin:xmax]
        if self.transform:
            img = self.transform(img)

        return img, target, negative_dict

    def __len__(self) -> int:
        return len(self.negative_list)


if __name__ == '__main__':
    root_dir = 'E:\\1_database\\VOC\\demo_svm'
    img_path = os.path.join(root_dir, 'JPEGImages')
    dataset = CustomSVMDataset(root_dir)

    negative_list = dataset.get_negatives()
    transform = dataset.get_transform()
    name_list = dataset.get_name_list()

    hard_negative_dataset = CustomHNMDataset(negative_list, name_list, transform=transform)
    image, target, negative_dict = hard_negative_dataset.__getitem__(100)

    print(target)
    print(negative_dict)
