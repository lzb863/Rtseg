import pandas as pd
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class VocDataset(Dataset):
    def __init__(self, cfg, split='train'):

        self.root = cfg.root
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.lbl_root = os.path.join(self.root, 'SegmentationClass')
        self.txt_root = os.path.join(self.root, 'ImageSets', 'Segmentation')

        # get txt
        assert split in ['train', 'trainval', 'val', 'test']
        self.train_set = np.array(pd.read_csv(os.path.join(self.txt_root, '{}.txt'.format(split)), names=['data'])['data'])
        self.split = split
        if self.split == 'train':
            self.transforms = cfg.train.transforms
        else:
            self.transforms = cfg.val.transforms

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        idx = 3
        im = Image.open(os.path.join(self.img_root, '{}.jpg'.format(self.train_set[idx])))
        if self.split == 'test':
            sample = {
                'img': im
            }
            sample = self.transforms(sample)
            sample['name'] = self.train_set[idx]
            return sample
        else:
            lb = Image.open(os.path.join(self.lbl_root, '{}.png'.format(self.train_set[idx])))
            sample = {
                'img': im,
                'lbl': lb
            }
            sample = self.transforms(sample)
            sample['name'] = self.train_set[idx]
            return sample
