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


def _find_config_file():
    filename = app_root + '/config/saved_temporal_model_defaults.json'
    if os.path.isfile(filename):
        return filename
    filename = osp.abspath(osp.join(osp.abspath(
        __file__), '../../config/saved_temporal_model_defaults.json'))
    if os.path.isfile(filename):
        return filename
    raise ValueError("failed_locate_config")

'''
run source/ronin_lstm_tcn.py with mode (train/test) and model type. 
Please refer to the source code for the full list of command line arguments. 
Optionally you can specify a configuration file such as config/temporal_model_defaults.json with the data paths.

Example training command: 
    python ronin_lstm_tcn.py train 
    --type tcn 
    --config <path-to-your-config-file> 
    --out_dir <path-to-output-folder> 
    --use_scheduler.

Example testing command: 
    python ronin_lstm_tcn.py test 
    --type tcn 
    --test_list <path-to-test-list> 
    --data_dir <path-to-dataset-folder> 
    --out_dir <path-to-output-folder> 
    --model_path <path-to-model-checkpoint>.

'''

def _get_args():
    """
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """

    default_config_file = _find_config_file()

    import argparse

    parser = argparse.ArgumentParser(description="Run seq2seq model in train/test mode [required]. Optional "
                                                 "configurations can be specified as --key [value..] pairs",
                                     add_help=True)
    parser.add_argument('--config', type=str, help='Configuration file [Default: {}]'.format(default_config_file),
                        default=default_config_file)
    # common
    parser.add_argument('--type', type=str,
                        choices=['tcn', 'lstm', 'lstm_bi'], help='Model type')
    parser.add_argument('--data_dir', type=str,
                        help='Directory for data files if different from list path.')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float,
                        help='Gaussian for smoothing features')
    parser.add_argument('--target_sigma', type=float,
                        help='Gaussian for smoothing target')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str,
                        help='Cuda device (e.g:- cuda:0) or cpu')
    parser.add_argument('--dataset', type=str, choices=['ronin', 'ridi'])
    # tcn
    tcn_cmd = parser.add_argument_group('tcn', 'configuration for TCN')
    tcn_cmd.add_argument('--kernel_size', type=int)
    tcn_cmd.add_argument('--channels', type=str,
                         help='Channel sizes for TCN layers (comma separated)')
    # lstm
    lstm_cmd = parser.add_argument_group('lstm', 'configuration for LSTM')
    lstm_cmd.add_argument('--layers', type=int)
    lstm_cmd.add_argument('--layer_size', type=int)

    mode = parser.add_subparsers(
        title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True
    # train
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str)
    train_cmd.add_argument('--val_list', type=str)
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int)
    train_cmd.add_argument('--save_interval', type=int)
    train_cmd.add_argument('--lr', '--learning_rate', type=float)
    # test
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_path', type=str, default=None)
    test_cmd.add_argument('--test_list', type=str, default=None)
    test_cmd.add_argument('--model_path', type=str, default=None)
    test_cmd.add_argument('--fast_test', action='store_true')
    test_cmd.add_argument('--show_plot', action='store_true')

    parser.add_argument('-f', type=str, default=None)
    
    '''
    Extra arguments
    Set True: use_scheduler, 
              quite (no output on stdout), 
              force_lr (force lr when a model is loaded from continue_from)
    float:  dropout, 
            max_ori_error (err. threshold for priority grv in degrees)
            max_velocity_norm (filter outliers in training) 
    '''

    args, unknown_args = parser.parse_known_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)

    print(args)
    return args, {}


def _fake_sys_argv():
    if "test" not in sys.argv:
        sys.argv.append("--mode")
        sys.argv.append("train")

def _fake_args(args):
    args.root_dir = app_root
    args.out_dir = app_root + "/output"
    args.train_list = app_root + "/lists/list_train_ridi_public.txt"
    args.dataset = "ridi"
    args.epochs = 2

    args.type = "tcn"
    args.data_dir = app_root 
    return args

def _prepare_args(new_args, arch="tcn"):
    _fake_sys_argv()
    args, kwargs = _get_args()
    args = _fake_args(args)
    #_run_test_body_heading(args, kwargs)

    if new_args is not None:
        for key, value in new_args.items():
            setattr(args, key, value)
    setattr(args, "arch", arch)
    setattr(args, "type", arch)
    return args


class RonninTcnTrain(object):
    @classmethod
    def train(cls, new_args):
        args = _prepare_args(new_args, arch="tcn")
        from source.ronin_lstm_tcn import train
        return train(args)

    @classmethod
    def test(cls, new_args):
        args = _prepare_args(new_args, arch="tcn")
        from source.ronin_lstm_tcn import test
        return test(args)

    @classmethod
    def select_model(cls, new_args):
        args = _prepare_args(new_args, arch="tcn")
        from source.ronin_lstm_tcn import get_model
        return get_model(args)

    @classmethod
    def inspect_model(cls, model):
        from source.ronin_lstm_tcn import inspect_model
        inspect_model(model)


class RonninLstmBiTrain(object):
    @classmethod
    def train(cls, new_args):
        args = _prepare_args(new_args, arch="lstm_bi")
        from source.ronin_lstm_tcn import train
        return train(args)

    @classmethod
    def test(cls, new_args):
        args = _prepare_args(new_args, arch="lstm_bi")
        from source.ronin_lstm_tcn import test
        return test(args)

    @classmethod
    def select_model(cls, new_args):
        args = _prepare_args(new_args, arch="lstm_bi")
        from source.ronin_lstm_tcn import get_model
        return get_model(args)

    @classmethod
    def inspect_model(cls, model):
        from source.ronin_lstm_tcn import inspect_model
        inspect_model(model)



class RonninLstmTrain(object):
    @classmethod
    def train(cls, new_args):
        args = _prepare_args(new_args, arch="lstm")
        from source.ronin_lstm_tcn import train
        return train(args)

    @classmethod
    def test(cls, new_args):
        args = _prepare_args(new_args, arch="lstm")
        from source.ronin_lstm_tcn import test
        return test(args)

    @classmethod
    def select_model(cls, new_args):
        args = _prepare_args(new_args, arch="lstm")
        from source.ronin_lstm_tcn import get_model
        return get_model(args)

    @classmethod
    def inspect_model(cls, model):
        from source.ronin_lstm_tcn import inspect_model
        inspect_model(model)


def _get_list_paths(list_path):
    test_paths = []
    with open(list_path, "rt") as f:
        for line in f.readlines():
            line = app_root + "/" + line.strip()
            if os.path.isdir(line):
                test_paths.append(line)
    return test_paths

if __name__ == '__main__':
    new_args = {}
    new_args["epochs"] = 1
    new_args["train_list"] = app_root + "/lists/list_train_ridi_tiny.txt"

    #new_args["model_path"] = app_root + "/trained_models/ronin_resnet/checkpoint_gsn_latest.pt"
    new_args["keep_training"] = True

    RonninKlass = RonninTcnTrain

    model = RonninKlass.select_model(new_args)
    RonninKlass.inspect_model(model)

    loss_v1, loss_v2, pt_path = RonninKlass.train(new_args)

    list_path = app_root + "/lists/list_test_ridi_tiny.txt"

    new_args["model_path"] = pt_path
    new_args["test_list"] = list_path
    new_args["test_path"] = None
    new_args["fast_test"] = False
    new_args["show_plot"] = True

    losses_avg = RonninKlass.test(new_args)
