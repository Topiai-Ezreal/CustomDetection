import numpy  as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from custom_finetune_data import CustomFinetuneDataset


class CustomBatchSampler(Sampler):
    def __init__(self, preset_num, num_positive, num_negative, batch_positive, batch_negative) -> None:
        """
        每次批量处理，其中batch_positive个正样本，batch_negative个负样本
        num_positive: 正样本数目
        num_negative: 负样本数目
        batch_positive: 单次正样本数
        batch_negative: 单次负样本数
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        self.idx_list = list(range(preset_num))
        self.batch = batch_negative + batch_positive
        self.num_iter = int(preset_num / self.batch) + 1

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter


class CustomBatchSampler_svm(Sampler):
    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        """
        每次批量处理，其中batch_positive个正样本，batch_negative个负样本
        num_positive: 正样本数目
        num_negative: 负样本数目
        batch_positive: 单次正样本数
        batch_negative: 单次负样本数
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))
        self.batch = batch_negative + batch_positive
        self.num_iter = int(length / self.batch) + 1

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter


def tes(idx):
    root_dir = 'E:\\1_database\\VOC\\demo_finetune'
    train_data_set = CustomFinetuneDataset(root_dir)
    preset_proposal_num = train_data_set.get_img_num() * 2000  # 预设的proposal数量，每张图2000个
    positive_num = train_data_set.get_positive_num()  # 所有的positive数量
    negative_num = train_data_set.get_negative_num()  # 所有的negative数量
    print(train_data_set.get_positive_num())
    print(train_data_set.get_negative_num())
    train_sampler = CustomBatchSampler(preset_proposal_num, positive_num, negative_num, 32, 96)

    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())
