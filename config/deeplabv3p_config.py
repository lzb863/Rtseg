from utils.transforms import RandomResizedCrop,RandomVerticalFlip, RandomHorizontalFlip,ToTensor, Normalize
from torchvision.transforms import Compose
from torch.utils.data import DistributedSampler


class Config:
    def __init__(self):
        self.root = '/root/voc/VOC2012'
        self.save_path = '/root/ckpts/'
        self.flag = 'deeplabv3plus'
        self.dataset = 'cityscapes' #'voc2012'
        self.is_training = True

        self.train = Config.Train()
        self.model = Config.Model()
        self.distributed = Config.Distributed()
        self.val = Config.Val()
        self.test = Config.Test()

    class Model:
        def __init__(self):
            self.output_stride = 16
            self.shortcut_dim = 48
            self.shortcur_kernel = 1
            self.aspp_out_dim = 256
            self.num_classes = 21

    class Train:
        def __init__(self):
            self.lr = 1e-5 #0.007
            self.lr_gamma = 0.1
            self.momentum = 0.9
            self.weight_decay = 0.00004
            self.bn_mom = 0.0003
            self.power = 0.9
            self.gpus = 1
            self.batch_size = 2
            self.epochs = 10
            self.eval_display = 200
            self.display = 2
            self.num_classes = 21
            self.ckpt_step = 5000
            self.workers = 1
            self.distributed = True
            self.crop_height = 512
            self.crop_width = 512
            self.sampler = DistributedSampler

            self.log_dir = './log'
            self.log_name = 'deeplabv3+'
            self.transforms = Compose([
                RandomResizedCrop(size=(self.crop_height, self.crop_width)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ])

    class Val:
        def __init__(self):
            self.model_path = ''
            self.batch_size = 1
            self.workers = 2
            self.result_dir = ''
            self.sampler = DistributedSampler
            self.transforms = Compose([
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ])

    class Test:
        def __init__(self):
            self.result_dir = './test'

    class Distributed:
        def __init__(self):
            self.backend = 'nccl'
            self.dist_url = 'tcp://127.0.0.1:12366'
            self.world_size = 1
            self.rank = 0
            self.gpu_id = 0
