from os import path as osp
import sys

import pandas
import numpy as np
import quaternion
from scipy._lib.doccer import inherit_docstring_from

from data_utils import CompiledSequence

import math


def euler_from_quaternion(x, y, z, w):
    """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

class VecRotator(object):
    @classmethod
    def get_init_ori_val_global(cls, tango_ori):
        # tango_pos_4 = tango_pos[0:4,:]
        # tango_pos_4_mean = np.mean(tango_pos_4, axis=0)
        # init_tango_ori = quaternion.quaternion(*tango_ori[0])
        # r_x, p_y, y_z = euler_from_quaternion(init_tango_ori.x, init_tango_ori.y, init_tango_ori.z, init_tango_ori.w)
        return tango_ori[0]

    @classmethod
    def get_ori_ref_global(cls, game_rv, init_tango_ori_val0):
        # Use game rotation vector as device orientation.
        init_tango_ori = quaternion.quaternion(*init_tango_ori_val0)

        init_game_rv = game_rv[0]
        init_rotor = init_tango_ori * init_game_rv.conj()
        ori = init_rotor * game_rv
        return ori

    @classmethod
    def rotate_xyz_globl(cls, var_xyz, ori):
        nz = np.zeros((var_xyz.shape[0], 1))
        var_xyz_q = quaternion.from_float_array(np.concatenate([nz, var_xyz], axis=1))
        var_xyz_r = quaternion.as_float_array(ori * var_xyz_q * ori.conj())[:, 1:]
        return var_xyz_r

    @classmethod
    def get_acce_global(cls, acce, ori):
        nz = np.zeros((acce.shape[0], 1))
        acce_q = quaternion.from_float_array(
            np.concatenate([nz, acce], axis=1))
        acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]
        return acce_glob


class RIDIGlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RIDI (can be downloaded from https://wustl.app.box.com/s/6lzfkaw00w76f8dmu0axax7441xcrzd9)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        self.info['ori_source'] = 'game_rv'

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            imu_all = pandas.read_pickle(osp.join(path, 'processed/data.pkl'))

        ts = imu_all[['time']].values / 1e09
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
    
        tango_pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values
        tango_ori = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)

        # Use game rotation vector as device orientation.
        # init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
        # game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)

        # init_rotor = init_tango_ori * game_rv[0].conj()
        # ori = init_rotor * game_rv
        ori_init_val0 = VecRotator.get_init_ori_val_global(tango_ori)
        ori = VecRotator.get_ori_ref_global(game_rv, ori_init_val0)
        #ori = VecRotator.get_ori_ref_global(game_rv)

        # convert to global frame -- both gyro and acc (rotated by ori)
        # nz = np.zeros(ts.shape)
        # gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        # gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
        gyro_glob = VecRotator.rotate_xyz_globl(gyro, ori)
        
        # nz = np.zeros(ts.shape)
        # acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
        # acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]
        acce_glob = VecRotator.rotate_xyz_globl(acce, ori)

        #shape [ns, 1]
        self.ts = ts
        #shape [ns, 6] -- combin gyro and acc together -- these data have been transformed to global frame
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        #shape [ns, 2]  -- speed for x, y only, not include z
        self.targets = (tango_pos[self.w:, :2] - tango_pos[:-self.w, :2]) / (ts[self.w:] - ts[:-self.w])
        
        # aux data -- just for reference
        self.gt_pos = tango_pos
        self.orientations = quaternion.as_float_array(game_rv)
        print("TS/Feature/Target shapes", self.ts.shape, self.features.shape, self.targets.shape, "GtPos/Ori shape", self.gt_pos.shape, self.orientations.shape, "Interval (samples/sec)", self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: orientation {}'.format(self.info['path'], self.info['ori_source'])
