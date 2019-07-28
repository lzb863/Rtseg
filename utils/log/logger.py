# From Gitlab http://192.168.193.253/gitlab/iccvdrones2019/dronesdet
import os
import torch
import torchvision.utils as vutils
import logging

from torch.utils.tensorboard import SummaryWriter
from utils.log.timer import Timer


class Logger(object):
    def __init__(self, cfg, main_processor_flag=True):
        self.main_processor_flag = main_processor_flag
        if self.main_processor_flag:
            self.log_dir = cfg.Train.log_dir
            if not os.path.exists(cfg.Train.log_dir):
                os.mkdir(cfg.Train.log_dir)
            self.tensorboard = SummaryWriter(os.path.join(cfg.Train.log_dir, cfg.Train.model_name))

        self.timer = Timer()
        self.max_iter_num = cfg.Train.max_iter_num
        self.timer.start(self.max_iter_num)

        self.logger = self._init_logger(cfg.Train.model_name)

    @staticmethod
    def _init_logger(logname):
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="\033[32;1m %(name)s\033[0m \033[31;1m%(module)s\033[0m : %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def add_scalar(self, data, tag, n_iter):
        self.tensorboard.add_scalar(tag, float(data), n_iter)

    def add_img(self, tag, data, n_iter):
        self.tensorboard.add_image(tag, data, n_iter)

    def write_log_file(self, text):
        with open(os.path.join(self.log_dir, 'log.txt'), 'a+') as writer:
            writer.write(text+'\n')

    def log(self, data, n_iter):
        """
        Print training log to terminal and save it to the log file.
        data is a dict like: {'scalar':[], 'imgs':[]}
        :param data: data to log.
        :param n_iter: current training step.
        :return: None
        """
        if self.main_processor_flag:
            log_str = "{} Iter. {}/{} | ".format(self.timer.stamp(n_iter), n_iter, self.max_iter_num)
            for k, v in data['scalar'].items():
                log_str += "{}: {:.4} ".format(k, float(v))
                self.add_scalar(float(v), tag=k, n_iter=n_iter)
            self.write_log_file(log_str)
            # self.logger.info(log_str)
            print(log_str)
            if 'imgs' in data:
                for k, v in data['imgs'].items():
                    vis_img = torch.cat(v, dim=0)
                    vis_img = vutils.make_grid(vis_img, normalize=True, scale_each=True)
                    self.add_img(k, vis_img, n_iter=n_iter)