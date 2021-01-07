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
    filename = app_root + '/config/heading_model_defaults.json'
    if os.path.isfile(filename):
        return filename
    filename = osp.abspath(osp.join(osp.abspath(__file__), '../../config/heading_model_defaults.json'))
    if os.path.isfile(filename):
        return filename
    raise ValueError("failed_locate_config")

def _fake_sys_argv():
    if "test" not in sys.argv:
        sys.argv.append("test")

def _get_args():
    """
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """
    default_config_file = _find_config_file()

    import argparse

    parser = argparse.ArgumentParser(description="Run seq2seq heading model in train/test mode [required]. Optional "
                                                 "configurations can be specified as --key [value..] pairs",
                                     add_help=True)
    parser.add_argument('--config', type=str, help='Configuration file [Default: {}]'.format(default_config_file),
                        default=default_config_file)
    # common
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
    parser.add_argument('--device', type=str, help='Cuda device e.g:- cuda:0')
    parser.add_argument('--cpu', action='store_const',
                        dest='device', const='cpu')
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
    test_cmd.add_argument('--prefix', type=str, default='',
                          help='prefix to add when saving files.')
    test_cmd.add_argument('--use_trajectory_type', type=str, default='gt',
                          help='Trajectory type to use when rendering the headings. (Default: gt). If not gt, the trajectory file is taken as <args.out_dir>/<data_name>_<use_trajectory_type>.npy with files generated in ronin_lstm_tcn.py or ronin_resnet.py')

    '''
    Extra arguments
    Set True: use_scheduler, quite (no output on stdout)
              force_lr (force lr when a model is loaded from continue_from),
              heading_norm (normalize heading),
              separate_loss (report loss separately for logging)
    float: dropout, max_ori_error (err. threshold for priority grv in degrees)
           max_velocity_norm (filter outliers in training)
           weights (array of float values) 
    '''
    args, unknown_args = parser.parse_known_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)
    if args.mode == "train" and kwargs.get('weights') and type(kwargs.get('weights')) != list:
        kwargs['weights'] = [float(i)
                             for i in kwargs.get('weights').split(',')]

    print(args, kwargs)
    return args, kwargs

def _run_test_body_heading(args, kwargs):
    from source.ronin_body_heading import test
    if not args.model_path:
        raise ValueError("Model path required")
    args.batch_size = 1
    test(args, **kwargs)

def _fake_args(args):
    args.model_path = app_root + "/trained_models/ronin_body_heading/checkpoints/ronin_body_heading.pt"
    args.test_path = app_root + "/ds_train_1/a000_1"
    return args

if __name__ == '__main__':
    _fake_sys_argv()
    args, kwargs = _get_args()
    args = _fake_args(args)
    _run_test_body_heading(args, kwargs)
