import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from PIL import Image
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from operators.base import BaseOperator

from models.fastscnn import Fast_SCNN

from utils.log.logger import Logger
from utils.dataset_tool import get_train_loader, get_val_loader
from utils.colormap import color_map
from utils.log.visualization import seg_vis


class FastSCNNOperator(BaseOperator):
    def __init__(self, cfg):
        super(FastSCNNOperator, self).__init__(cfg)
        # prepare model for train
        self.model = Fast_SCNN(input_channel=3, num_classes=self.cfg.Train.num_classes).cuda(self.cfg.Distributed.gpu_id)

        if self.cfg.Distributed.dist:
            self.model = DistributedDataParallel(self.model, find_unused_parameters=True, device_ids=[self.cfg.Distributed.gpu_id])

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg.Train.learning_rate,
                                          weight_decay=self.cfg.Train.weight_decay)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.9, momentum=0.9)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=255)

        if self.cfg.is_training:
            self.loader = get_train_loader(name=self.cfg.Dataset.name, cfg=self.cfg)
        else:
            self.loader = get_val_loader(name=self.cfg.Dataset.name, cfg=cfg)
        self.cfg.Train.max_iter_num = self.cfg.Train.epochs * len(self.loader)

        self.main_flag = self.cfg.Distributed.gpu_id == 0

        if self.model.training:
            self.logger = Logger(self.cfg, self.main_flag)

    def adjust_lr(self, itr, max_itr):
        now_lr = self.cfg.Train.learning_rate * (1 - itr / (max_itr + 1)) ** self.cfg.Train.power
        self.optimizer.param_groups[0]['lr'] = now_lr
        # self.optimizer.param_groups[1]['lr'] = 10 * now_lr
        return now_lr

    def criterion(self, outs, labels):
        return self.loss(outs, labels)

    @staticmethod
    def save_ckp(models, step, path):
        """
        Save checkpoint of the model.
        :param models: nn.Module
        :param step: step of the checkpoint.
        :param path: save path.
        """
        torch.save(models.state_dict(), os.path.join(path, 'ckp-{}.pth'.format(step)))

    def training_process(self):
        logger = Logger(self.cfg, self.main_flag)
        self.model.train()
        colormap = color_map[self.cfg.Dataset.name]

        itr = 0

        for epoch in range(self.cfg.Train.epochs):

            for idx, sample in enumerate(self.loader):

                newlr = self.adjust_lr(itr=itr, max_itr=self.cfg.Train.max_iter_num)

                xs,ys, names = sample

                ys = torch.squeeze(ys, dim=1)
                xs = xs.cuda(self.cfg.Distributed.gpu_id)
                ys = ys.cuda(self.cfg.Distributed.gpu_id)
                preds = self.model(xs)
                loss = self.criterion(preds, ys.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_img = seg_vis(preds, colormap)
                gt_img = seg_vis(ys, colormap)

                pred_img = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.

                if self.main_flag:
                    if itr % self.cfg.Train.print_steps == self.cfg.Train.print_steps - 1:
                        log_data = {
                            'scalar': {
                                'Loss': loss.item()
                            },
                            'imgs': {
                                'Pred': [pred_img, gt_img]
                            }
                        }
                        logger.log(log_data, n_iter=itr)

                itr += 1

                if itr % self.cfg.Train.save_model == self.cfg.Train.save_model - 1:
                    if not os.path.exists(self.cfg.Train.ckp_dir):
                        os.mkdir(self.cfg.Train.ckp_dir)
                        if not os.path.exists(os.path.join(self.cfg.Train.ckp_dir, self.cfg.Train.model_name)):
                            os.mkdir(os.path.join(self.cfg.Train.ckp_dir, self.cfg.Train.model_name))
                    self.save_ckp(self.model.module, itr,
                                  os.path.join(self.cfg.Train.ckp_dir, self.cfg.Train.model_name))

    def eval_process(self):
        self.model.eval()
        self.model.module.load_state_dict(torch.load(self.cfg.Val.model_file))
        epoch = 0
        step = 0
        class_converter = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]).astype(
            np.uint8)

        self.loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            for sample in self.loader:
                xs, ys, names = sample

                xs = xs.cuda(self.cfg.Distributed.gpu_id)
                outs = self.model(xs)
                pred_img = torch.max(outs, dim=1)[1].cpu()
                pred_img = pred_img.numpy().astype(np.uint8)

                print(pred_img.max())
                print(pred_img.shape)
                for i in range(pred_img.shape[0]):
                    img = class_converter[pred_img[i]]
                    # pred_img = class_converter[pred_img]
                    im = Image.fromarray(img)
                    img_dir = names[i].split(sep='/')[-2]
                    name = '_'.join(names[i].split(sep='/')[-1].split(sep='_')[:-1]) + '.png'

                    if not os.path.exists(os.path.join(self.cfg.Val.result_dir, img_dir)):
                        os.mkdir(os.path.join(self.cfg.Val.result_dir, img_dir))
                    im.save(os.path.join(self.cfg.Val.result_dir, img_dir, name))
                if self.main_flag:
                    step += 1
                    print('Step : %d / %d' % (step, len(self.loader)))
            print('Done !!')


if __name__ == '__main__':
    import os
    import sys

    sys.path.append('../')
    from config.defaults import config
    from operators.dist_wrapper import DistributedWrapper

    trainer = DistributedWrapper(config, FastSCNNOperator)
    trainer.train()
    # trainer.eval()

