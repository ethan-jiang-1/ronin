import os
import sys
app_root = os.path.dirname(os.path.dirname(__file__))
print(app_root)
sys.path.append(app_root)
sys.path.append(app_root + "/source")

from source.data_ridi import RIDIGlobSpeedSequence
from source.data_glob_speed import SequenceToSequenceDataset
from source.utils import load_config


def get_args():
    from os import path as osp
    import numpy as np

    """
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """
    default_config_file = osp.abspath(osp.join(osp.abspath(
        __file__), '../../config/temporal_model_defaults.json'))

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

    print(args, kwargs)
    if args.mode == 'train':
        #train(args, **kwargs)
        pass
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        #test(args, **kwargs)
        pass
    return args, kwargs

def alter_argv():
    import sys
    print(sys.argv)
    sys.argv += ["--mode", "train"]


def read_dataset_rdii_by_loader():
    alter_argv()
    args, kwargs = get_args()
    print("args", args)
    print("kwargs", kwargs)


    seq_type = RIDIGlobSpeedSequence


    root_dir = app_root + "/ds_rdii"
    data_list = ["dan_bag1"]

    random_shift = False
    transforms = None
    shuffle = False
    grv_only = False

    dataset = SequenceToSequenceDataset(seq_type, root_dir, data_list, args.cache_path, 
                                        args.step_size, args.window_size,
                                        random_shift=random_shift, transform=transforms, 
                                        shuffle=shuffle,
                                        grv_only=grv_only, 
                                        **kwargs)

    print(dataset)


def read_dataset_rdii_directly(w=1):
    import pandas as pd
    import quaternion
    import numpy as np

    data_name = "dan_bag1"    
    csv_path = app_root + "/ds_rdii/{}/processed/data.csv".format(data_name)

    imu_all = pd.read_csv(csv_path)
    print(imu_all)

    ts = imu_all[['time']].values / 1e09
    gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
    acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
    tango_pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values

    tango_ori = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    rv = imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values

    init_tango_ori = quaternion.quaternion(*tango_ori[0])
    game_rv = quaternion.from_float_array(rv)

    init_rotor = init_tango_ori * game_rv[0].conj()
    ori = init_rotor * game_rv

    nz = np.zeros(ts.shape)
    gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

    gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
    acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

    #w = 1 # interval
    features = np.concatenate([gyro_glob, acce_glob], axis=1)
    targets = (tango_pos[w:, :2] - tango_pos[:-w, :2]) / (ts[w:] - ts[:-w])

    return features, targets

read_dataset_rdii_directly()
