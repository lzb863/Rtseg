from dataset.vocdtset import VocDataset
from torch.utils.data import DataLoader
import numpy as np
from dataset.cityscapes import CityScapes
from torch.utils.data import DistributedSampler



def get_train_loader(name, cfg):
    if name == 'voc2012':
        dataset = VocDataset(cfg=cfg, split='train')
    elif name == 'cityscapes':
        dataset = CityScapes(cfg=cfg, split='train')
    else:
        dataset = VocDataset(cfg=cfg, split='train')
    return make_dataloader(dataset, cfg)


def get_val_loader(name, cfg):
    if name == 'voc2012':
        dataset = VocDataset(cfg=cfg, split='val')
    elif name == 'cityscapes':
        dataset = CityScapes(cfg=cfg, split='val')
    else:
        dataset = VocDataset(cfg=cfg, split='val')
    return make_dataloader(dataset, cfg)


def make_dataloader(dataset, cfg):
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                        batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                        sampler=sampler if sampler else None,
                        pin_memory=True, shuffle=True if sampler is None else False)
    return loader


def voc_colormap(N=256):

    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        cmap[i, :] = [r, g, b]
    return cmap
