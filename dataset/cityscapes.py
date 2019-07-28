from dataset.transforms import Compose,RandomVerticalFlip,RandomHorizontalFlip,ToTensor,RandomCrop,Normalize,CalsBorderMask
from torchvision.datasets.cityscapes import Cityscapes
from torch.utils.data import Dataset


class CityScapes(Dataset):
    def __init__(self, cfg, split):
        super(CityScapes, self).__init__()
        self.cfg = cfg
        self.dataset = Cityscapes(cfg.Dataset.root, split=split, mode='fine', target_type='semantic')

        self.transforms = build_transforms(self.cfg, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx = 3
        image, smnt = self.dataset[idx]
        name = self.dataset.images[idx]
        if self.transforms is not None:
            image, smnt = self.transforms(image, smnt)
        return image, smnt, name
        #     image, smnt = self.transforms(image, smnt)
        # return image, smnt, name


def build_transforms(cfg, split='train'):
    norm = Normalize(mean=cfg.Train.PIL_input_mean, std=cfg.Train.PIL_input_std)
    if split == 'train':
        transforms = Compose([
            RandomCrop(size=cfg.Train.crop_size),
            # RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip(p=0.5),
            ToTensor(),
            norm,
            # CalsBorderMask()
        ])
    else:
        transforms = Compose([
            ToTensor(),
            norm
        ])

    return transforms
