import numpy as np

color_map = {
    'cityscapes': np.array([
        [128, 64, 128],
        [244,35,232],
        [70,70,70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170,30],
        [220,220,0],
        [107,142,35],
        [152,251,152],
        [70,130,180],
        [220,20,60],
        [255,0,0],
        [0,0,142],
        [0,0,70],
        [0,60,100],
        [0,80,100],
        [0,0,230],
        [119,11,32],
        [0,0,0]
    ]),
    'ins_city': [
        [0,0,0],
        [220,20,60],
        [255,0,0],
        [0,0,142],
        [0,0,70],
        [0,60,100],
        [0,80,100],
        [0,0,230],
        [119,11,32]
    ]
}


class_names = {
    'cityscapes': np.array([
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'trafficLight',
        'trafficSign',
        'vegetation',
        'terrain',
        'sky',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
        'background'
    ]),
    'ins_city': [
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle'
    ]
}