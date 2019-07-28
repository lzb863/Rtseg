from easydict import EasyDict as edict

config = edict()
config.is_training = True #False #True False

config.Train = edict()
config.Train.log_dir = './log'
config.Train.model_name = 'Fastscnn'
config.Train.seed = 333
config.Train.num_classes = 19
config.Train.crop_size = (512, 1024)
config.Train.max_iter_num = 90000

config.Train.min_size_train = (800, )
config.Train.max_size_train = 1024


# if we read data from cv2 to train our network, we should set the `config.Train.to_bgr255` to True
# if we read data from PIL to train out network, just keep it False
config.Train.to_bgr255 = False

# Transforms
config.Train.PIL_input_mean = [0.485, 0.456, 0.406]
config.Train.PIL_input_std = [0.229, 0.224, 0.225]

config.Train.CV2_input_mean = [102.9801, 115.9465, 122.7717]
config.Train.CV2_input_std = [1., 1., 1.]

# Image ColorJitter
config.Train.brightness = 0.0
config.Train.contrast = 0.0
config.Train.saturation = 0.0
config.Train.hue = 0.0

config.Train.vertical_flip_prob_train = 0.0
config.Train.horizontal_flip_prob_train = 0.0

# output stride or downsampling times
config.Train.size_divisibility = 32

# Batch size
config.Train.batch_size = 8

# Num workers
config.Train.num_workers = 4

# Learning rate
config.Train.learning_rate = 0.001 # 1e-4
config.Train.weight_decay = 0.0001
config.Train.momentum = 0.9
config.Train.power = 0.9

config.Train.epochs = 300

config.Train.lr_steps = (30000,)
config.Train.lr_gamma = 0.1

config.Train.warmup_factor = 1.0 / 3
config.Train.warmup_iters = 500
config.Train.warmup_method = 'linear'

# Max iter
config.Train.max_iter_num = 90000
config.Train.lr_milestones = [60000, 80000]
config.Train.save_model = 10000
config.Train.ckp_dir = './ckp'

# Print steps
config.Train.print_steps = 10

# Model -----------------------
# -----------------------------


# Distributed
config.Distributed = edict()
config.Distributed.dist = True # False
config.Distributed.backend = 'nccl'
config.Distributed.rank = 0
config.Distributed.gpu_id = -1
config.Distributed.world_size = 1
config.Distributed.ngpus_per_node = 1
config.Distributed.dist_url = 'tcp://127.0.0.1:34567'

# Dataset
config.Dataset = edict()
config.Dataset.name = 'cityscapes'
config.Dataset.root = '/root/city'#'./data/cityscapes'

# Val
config.Val = edict()
config.Val.model_file = './log/ckp-9999.pth'  # normal-29999.pth
config.Val.result_dir = './log/results/'

config.Val.confidence_threshold = 0.7



# Test ----------

config.Test = edict()
config.Test.batch_size = 1

config.Test.min_size_train = (800, )
config.Test.max_size_train = 1333