import numpy as np
import math


class TrajOdometer(object):

    @classmethod
    def get_rmse(cls, traj_gt, traj_pr):
        #shape of traj is (n, 3): n points x, y, z
        com_len = min(traj_gt.shape[0], traj_pr.shape[0])
        traj_gt = traj_gt[0:com_len:, :]
        traj_pr = traj_pr[0:com_len:, :]

        return np.sqrt(np.mean(np.square(np.linalg.norm(traj_gt - traj_pr, axis=-1))))

    @classmethod
    def get_distance_3D(cls, traj_x):
        distance_all = 0
        for i in range(1, len(traj_x)):
            p0 = traj_x[i - 1]
            p1 = traj_x[i]
            d = math.sqrt((p1[0] - p0[0]) ** 2 +
                          (p1[1] - p0[1]) ** 2 +
                          (p1[2] - p0[2]) ** 2)
            distance_all += d
        return distance_all

    @classmethod
    def get_distance_2D(cls, traj_x):
        distance_all = 0
        for i in range(1, len(traj_x)):
            p0 = traj_x[i - 1]
            p1 = traj_x[i]
            d = math.sqrt((p1[0] - p0[0]) ** 2 +
                          (p1[1] - p0[1]) ** 2)
            distance_all += d
        return distance_all

    @classmethod
    def get_speed_3D(cls, traj_x, stride=10):
        distance_all = 0
        len_traj = len(traj_x)

        time_span_pt = 0.1 * (10 / stride)
        time_span_all = len_traj * time_span_pt

        sps = np.zeros((len_traj - 1))
        for i in range(1, len_traj):
            p0 = traj_x[i - 1]
            p1 = traj_x[i]
            d = math.sqrt((p1[0] - p0[0]) ** 2 +
                          (p1[1] - p0[1]) ** 2 +
                          (p1[2] - p0[2]) ** 2)
            distance_all += d
            sps[i - 1] = d / time_span_pt
        speed = distance_all / time_span_all
        return speed, time_span_all, distance_all, sps
