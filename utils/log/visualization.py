import torch
import numpy as np

from torch.nn.functional import softmax


def seg_vis(tensor, color_map):
    if tensor.dtype == torch.uint8:
        img = tensor[0].cpu().numpy()
        img[img == 255] = 19
        img = color_map[img].astype(np.uint8)
    else:
        img = softmax(tensor[0], dim=0)
        img = torch.max(img, dim=0)[1]
        img = color_map[img.cpu().numpy()].astype(np.uint8)
    return img