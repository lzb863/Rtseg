class BaseOperator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def criterion(self, outs, labels):
        raise NotImplementedError

    def training_process(self):
        raise NotImplementedError

    def eval_process(self):
        raise NotImplementedError