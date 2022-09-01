"""
Utilities of Project
"""

import argparse

from yaml import parse


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(
        self,
        start_val=0,
        start_count=0,
        start_avg=0,
        start_sum=0
    ):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
            Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
            Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def get_args():
    """
    Argunments of:
        - `train.py`
        - `test.py`
    """
    parser = argparse.ArgumentParser(
        description='Arguemnt Parser of `Train` and `Evaluation` of deep neural network.')

    # Hardware
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                        default=True, help='Use GPU')
    parser.add_argument('-w', '--num-workers', dest='num_workers', default=1,
                        type=int, help='Number of workers for loading data')

    # CNN Backbone
    parser.add_argument('--backbone', dest='backbone', default="resnet34",
                        type=str, help="Network Backbones: ['NIN', 'resnet34']")

    # Data Path
    # - Train
    parser.add_argument('--train-data-path', dest='train_data_path', default="./datasets/cifar10",
                        type=str, help='train dataset base directory')
    # - validation
    parser.add_argument('--val-data-path', dest='val_data_path', default="./datasets/cifar10",
                        type=str, help='validation dataset base directory')
    # - Test
    parser.add_argument('--test-data-path', dest='test_data_path', default="./datasets/cifar10",
                        type=str, help='Test dataset base directory')

    # Dataloader Parameters
    parser.add_argument('--show-all-angles', action='store_true', dest='show_all_angles',
                        default=False, help='Giving all diferent angles to the model.')
    parser.add_argument('--shuffle-data', action='store_true', dest='shuffle_data',
                        default=False, help='Shuffle data')

    # Model Parameters
    parser.add_argument("--feature-layer-index", dest="feature_layer_index", default=1, type=int,
                        help="feature layer index that is used to extract feature for resnet architecture. valid value: [1,2,3,4].")

    # Optimizer Parameters
    parser.add_argument("--optimizer", dest="optimizer", default="adam", type=str,
                        help="Optimization Algorithm")
    parser.add_argument("--num-epochs", dest="num_epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch-size", dest="batch_size", default=64, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--lr", dest="lr", default=4.8,
                        type=float, help="learning rate")
    parser.add_argument("--momentum", dest="momentum", default=0.9, type=float,
                        help="momentum of SGD algorithm")
    parser.add_argument("--weight-decay", dest="weight_decay",  default=1e-6,
                        type=float, help="weight decay")

    # Saving Parameters
    parser.add_argument("--ckpt-save-path", dest="ckpt_save_path", type=str, default="./checkpoints",
                        help="Checkpoints address for saving")
    parser.add_argument("--ckpt-load-path", dest="ckpt_load_path", type=str, default=None,
                        help="Checkpoints address for loading")
    parser.add_argument("--ckpt-prefix", dest="ckpt_prefix", type=str, default="ckpt_",
                        help="Checkpoints prefix for saving a checkpoint")
    parser.add_argument("--ckpt-save-freq", dest="ckpt_save_freq", type=int, default=20,
                        help="Saving checkpoint frequency")
    parser.add_argument("--report-path", dest="report_path", type=str, default="./reports",
                        help="Saving report path")

    options = parser.parse_args()

    return options
