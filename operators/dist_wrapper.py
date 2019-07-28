import torch
import torch.multiprocessing as np
import torch.distributed as dist


class DistributedWrapper(object):
    def __init__(self, cfg, operator_class):
        self.cfg = cfg
        self.operator_class = operator_class

    def train(self):
        self.setup_params()
        np.spawn(self.dist_train_process, nprocs=self.cfg.Distributed.ngpus_per_node,
                 args=(self.cfg.Distributed.ngpus_per_node, self.cfg))

    def eval(self):
        self.setup_params()
        np.spawn(self.dist_eval_process, nprocs=self.cfg.Distributed.ngpus_per_node,
                 args=(self.cfg.Distributed.ngpus_per_node, self.cfg))

    def setup_params(self):
        ngpus_per_node = torch.cuda.device_count()
        self.cfg.Distributed.ngpus_per_node = ngpus_per_node
        self.cfg.Distributed.world_size = ngpus_per_node * self.cfg.Distributed.world_size

    def dist_train_process(self, gpu, ngpus_per_node, cfg):
        operator = self.init_operator(gpu, ngpus_per_node, cfg)
        operator.training_process()

    def dist_eval_process(self, gpu, ngpus_per_node, cfg):
        operator = self.init_operator(gpu, ngpus_per_node, cfg)
        operator.eval_process()

    def init_operator(self, gpu, ngpus_per_node, cfg):
        cfg.Distributed.gpu_id = gpu
        print('Using GPU: %d' % gpu)
        cfg.Distributed.rank = cfg.Distributed.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.Distributed.backend, init_method=cfg.Distributed.dist_url,
                                world_size=cfg.Distributed.world_size, rank=cfg.Distributed.rank)
        torch.cuda.set_device(gpu)
        return self.operator_class(cfg)