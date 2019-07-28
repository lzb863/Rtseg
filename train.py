from config.defaults import config
from expr.fastscnn_op import FastSCNNOperator
# from expr.fastscnnBorder_op import FastSCNNOperator
from operators.dist_wrapper import DistributedWrapper


if __name__ == '__main__':
    trainer = DistributedWrapper(config, FastSCNNOperator)
    trainer.train()