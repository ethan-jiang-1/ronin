# hack to hook modules in subfolder

import numpy as np

import os
import sys
app_root = os.path.dirname(os.path.dirname(__file__))
app_root_source = app_root + "/source"
app_root_exam = app_root + "/exam"
if app_root not in sys.path:
    sys.path.append(app_root)
if app_root_source not in sys.path:
    sys.path.append(app_root_source)
if app_root_exam not in sys.path:
    sys.path.append(app_root_exam)

from os import path as osp
from source.utils import load_config


print("app_root", app_root)


'''
To train/test RoNIN ResNet model:
run source/ronin_resnet.py with mode argument. Please refer to the source code for the full list of command line arguments.
Example training command: 
    python ronin_resnet.py 
        --mode train 
        --train_list <path-to-train-list> 
        --root_dir <path-to-dataset-folder> 
        --out_dir <path-to-output-folder>.

Example testing command: 
    python ronin_resnet.py 
        --mode test 
        --test_list <path-to-train-list> 
        --root_dir <path-to-dataset-folder> 
        --out_dir <path-to-output-folder> 
        --model_path <path-to-model-checkpoint>.

'''

def _get_args():
    """
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    parser.add_argument('--model_summary', type=bool, default=True)

    parser.add_argument('-f', type=str, default=None)
    args = parser.parse_args()

    print(args)
    return args, {}

# def _run_test_body_heading(args, kwargs):
#     from source.ronin_body_heading import test
#     if not args.model_path:
#         raise ValueError("Model path required")
#     args.batch_size = 1
#     test(args, **kwargs)


def _run_train_resnet(args, kwargs):
    from source.ronin_resnet import train
    return train(args)


def _fake_sys_argv():
    if "test" not in sys.argv:
        sys.argv.append("--mode")
        sys.argv.append("train")

def _fake_args(args):
    #args.model_path = app_root + "/trained_models/ronin_resnet/checkpoint_gsn_latest.pt"
    args.root_dir = app_root
    #args.test_path = app_root + "/ds_train_1/a001_1"
    args.out_dir = app_root + "/output"
    args.train_list = app_root + "/lists/list_train_ridi_public.txt"
    args.dataset = "ridi"
    args.epochs = 2
    return args

def _train(new_args):
    _fake_sys_argv()
    args, kwargs = _get_args()
    args = _fake_args(args)
    #_run_test_body_heading(args, kwargs)

    if new_args is not None:
        for key, value in new_args.items():
            setattr(args, key, value)

    return _run_train_resnet(args, kwargs)


class RonninResnetTrain(object):
    @classmethod
    def train(cls, new_args):
        return _train(new_args)



if __name__ == '__main__':
    new_args = {}
    new_args["epochs"] = 100
    RonninResnetTrain.train(new_args)

