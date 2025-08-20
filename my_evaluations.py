
import json
import os
import shutil
import argparse
import scipy.stats
import numpy as np
import pandas as pd
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from utils import get_gps, read_data_from_file, read_data_from_file2 ,read_logs_from_file ,get_logger
from geopy import distance
from tqdm import tqdm
import hausdorff
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev

import experiment
from experiment import SquareQuery ,RoadQuery
# from scipy.spatial import distance
# from scipy.spatial import minkowski_distance
# import seaborn as sns


from fastdtw import fastdtw
import math
from math import sin, cos, asin, sqrt, radians
import numba
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method1
# import traj_dist.distance as tdist


# def hausdorff_metric(truth, pred, distance='haversine'):
#     """豪斯多夫距离
#     ref: https://github.com/mavillan/py-hausdorff
#
#     Args:
#         truth: 经纬度点，(trace_len, 2)
#         pred: 经纬度点，(trace_len, 2)
#         distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine
#
#     Returns:
#
#     """
#     return hausdorff.hausdorff_distance(truth, pred, distance=distance)
#
#
# def haversine(array_x, array_y):
#     R = 6378.0
#     radians = np.pi / 180.0
#     lat_x = radians * array_x[0]
#     lon_x = radians * array_x[1]
#     lat_y = radians * array_y[0]
#     lon_y = radians * array_y[1]
#     dlon = lon_y - lon_x
#     dlat = lat_y - lat_x
#     a = (pow(math.sin(dlat/2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon/2.0), 2.0))
#     return R * 2 * math.asin(math.sqrt(a))
#
#
# def dtw_metric(truth, pred, distance='haversine'):
#     """动态时间规整算法
#     ref: https://github.com/slaypni/fastdtw
#
#     Args:
#         truth: 经纬度点，(trace_len, 2)
#         pred: 经纬度点，(trace_len, 2)
#         distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine
#
#     Returns:
#
#     """
#     if distance == 'haversine':
#         distance, path = fastdtw(truth, pred, dist=haversine)
#     elif distance == 'manhattan':
#         distance, path = fastdtw(truth, pred, dist=cityblock)
#     elif distance == 'euclidean':
#         distance, path = fastdtw(truth, pred, dist=euclidean)
#     elif distance == 'chebyshev':
#         distance, path = fastdtw(truth, pred, dist=chebyshev)
#     elif distance == 'cosine':
#         distance, path = fastdtw(truth, pred, dist=cosine)
#     else:
#         distance, path = fastdtw(truth, pred, dist=euclidean)
#     return distance
#
#
rad = math.pi / 180.0
R = 6378137.0
#
def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)

    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point

    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def s_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    # C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr

def edit_distance(trace1, trace2):
    trace1 = np.array(trace1).astype(np.int32)
    trace2 = np.array(trace2).astype(np.int32)
    return edit_distance_impl(trace1, trace2)


@numba.jit(numba.int32(numba.int32[:], numba.int32[:]), nopython=True, nogil=True, cache=True)
def edit_distance_impl(trace1, trace2):
    dp = np.zeros((2, len(trace2) + 1), dtype=np.int32)
    for i, s1 in enumerate(trace1):
        for j, s2 in enumerate(trace2):
            d = 1 if s1 != s2 else 0
            dp[(i + 1) % 2, j + 1] = min(
                dp[i % 2, j + 1] + 1,
                dp[(i + 1) % 2, j] + 1,
                dp[i % 2, j] + d,
                )
    return dp[len(trace1) % 2, len(trace2)]


def hausdorff_metric(truth, pred, distance='haversine'):
    """hausdorff distance
    ref: https://github.com/mavillan/py-hausdorff
    Args:
        truth: longitude and latitude, (trace_len, 2)
        pred: longitude and latitude, (trace_len, 2)
        distance: computation method for distance, include haversine, manhattan, euclidean, chebyshev, cosine
    """
    return hausdorff.hausdorff_distance(truth, pred, distance=distance)


@numba.jit(numba.float64(numba.float64[:], numba.float64[:]), nopython=True, nogil=True, cache=True)
def haversine(array_x, array_y):
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = sin(dlat / 2) ** 2 + cos(lat_x) * cos(lat_y) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371


def dtw_metric(truth, pred, distance='haversine'):
    """ dynamic time wrapping
    ref: https://github.com/slaypni/fastdtw
    Args:
        truth: longitude and latitude, (trace_len, 2)
        pred: longitude and latitude, (trace_len, 2)
        distance: computation method for distance, include haversine, manhattan, euclidean, chebyshev, cosine
    """
    if distance == 'haversine':
        distance, path = fastdtw(truth, pred, dist=haversine)
    elif distance == 'manhattan':
        distance, path = fastdtw(truth, pred, dist=cityblock)
    elif distance == 'euclidean':
        distance, path = fastdtw(truth, pred, dist=euclidean)
    elif distance == 'chebyshev':
        distance, path = fastdtw(truth, pred, dist=chebyshev)
    elif distance == 'cosine':
        distance, path = fastdtw(truth, pred, dist=cosine)
    else:
        distance, path = fastdtw(truth, pred, dist=euclidean)
    return distance


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    # ca[i,j]之前计算过
    elif i == 0 and j == 0:
        ca[i, j] = haversine(P[0], Q[0])
        # 刚刚考虑P序列的0和Q序列的0时
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, P, Q), haversine(P[i], Q[0]))
    # 刚刚考虑P序列的i和Q序列的0时
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, P, Q), haversine(P[0], Q[j]))
        # 刚刚考虑P序列的0和Q序列的j时
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i - 1, j, P, Q),  # P动Q不动
                           _c(ca, i - 1, j - 1, P, Q),  # P不动Q动
                           _c(ca, i, j - 1, P, Q)),  # 一起动
                       haversine(P[i], Q[j]))
        # min是不考虑i，j时，至少需要多长的”狗绳“
        # 再取max表示考虑i，j时，需要多长的”狗绳“
    else:
        ca[i, j] = float("inf")
        # 非法的无效数据，算法中不考虑，此时 i<0,j<0
    return ca[i, j]

def Frechet_distance(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    # ca初始化成全-1的矩阵，shape = ( len(a), len(b) )
    dis = _c(ca, len(P) - 1, len(Q) - 1, P, Q)
    return dis

def g_lcss(t0, t1,eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1+1) for _ in range(n0+1)]
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            if great_circle_distance(t0[i-1,0],t0[i-1,1],t1[j-1,0],t1[j-1,1])<eps:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    lcss = 1-float(C[n0][n1])/min([n0,n1])
    return lcss

class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod  # 静态方法通过类名直接调用，不需要实例
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0., 1. / bins))
        ret_dist, ret_base = [], []
        for i in range(bins):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum() + 1e-14)
        p2 = p2 / (p2.sum() + 1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js


class IndividualEval(object):

    def __init__(self, data):

        if data == "BJ_Taxi":
            self.max_locs = 37684
            self.max_time = 2880
            self.max_distance2 = 0.00041326627851572266
            self.max_radius = 0.005635
            self.max_len = 60
            self.top_k_G = 350
            self.top_k_I = 50

            self.data = "BJ_Taxi"
            self.lon_0 = 116.25
            self.lat_0 = 39.79
            self.lon_range = 0.2507
            self.lat_range = 0.21  # span of latitude
            self.img_unit = 0.006  # grid size like 0.42 km * 0.55 km
            self.img_width = math.ceil(self.lon_range / self.img_unit) + 1
            self.img_height = math.ceil(self.lat_range / self.img_unit) + 1
            self.query_num = 1000
            self.pattern_num = 500

            self.X, self.Y = get_gps('./data/{}/gps.data'.format(data))
            with open("data/{}/bj_rid_gps.json".format(self.data), "r") as f:
                self.road_gps = json.load(f)

        elif data == "Porto_Taxi":
            self.X, self.Y = get_gps('./data/{}/gps.data'.format(data))  # X是纬度,Y是经度
            self.max_locs = 10904
            self.max_distance2 = 0.00014
            self.max_radius = 0.00190955
            self.max_len = 173
            self.top_k_G = 350
            self.top_k_I = 50
            self.max_time = 2880

            self.data = "Porto_Taxi"
            self.lon_0 = -8.6887  # min
            self.lat_0 = 41.1405  # min
            self.lon_1 = -8.5557  # max
            self.lat_1 = 41.1865  # max
            self.lon_range = 0.133
            self.lat_range = 0.046
            self.img_unit = 0.003   #0.0035
            self.img_width = math.ceil(self.lon_range / self.img_unit) + 1
            self.img_height = math.ceil(self.lat_range / self.img_unit) + 1
            self.query_num=500
            self.pattern_num=300

            with open("data/{}/porto_rid_gps.json".format(self.data), "r") as f:
                self.road_gps = json.load(f)


    def get_topk_visits(self, trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            for i in range(len(topk), k):
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / len(traj)
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)

    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_locs, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)

    def get_near_distances(self, trajs):  # 行驶距离
        distances = []
        for traj in trajs:
            for i in range(len(traj) - 1):
                dx = self.X[traj[i]] - self.X[traj[i + 1]]
                dy = self.Y[traj[i]] - self.Y[traj[i + 1]]
                distances.append(dx ** 2 + dy ** 2)
        print(max(distances))
        distances = np.array(distances, dtype=float)
        return distances

    def get_total_distances(self, trajs):  # 计算总的行驶距离
        distances = []

        for traj in tqdm(trajs, desc="Calculate Distance"):
            travel_distance = 0  # 每条轨迹的长度
            for i in range(len(traj) - 1):
                pre_gps_x1 = self.X[traj[i]]
                pre_gps_y1 = self.Y[traj[i]]
                gps_x2 = self.X[traj[i + 1]]
                gps_y2 = self.Y[traj[i + 1]]
                travel_distance += distance.distance((gps_x2, gps_y2), (pre_gps_x1, pre_gps_y1)).kilometers
            distances.append(travel_distance)

        print(max(distances))
        distances = np.array(distances, dtype=float)
        return distances

    def get_gradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :return:
        """
        gradius = []
        for traj in tqdm(trajs, desc="Calculate radius"):
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(traj)):
                lat = xs[i]
                lng = ys[i]
                dis = distance.distance((xcenter, ycenter), (lat, lng)).kilometers
                rad.append(dis)
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        print(np.max(gradius))
        return gradius

    def get_radius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in tqdm(trajs, desc="Calculate radius"):
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [dxs[i] ** 2 + dys[i] ** 2 for i in range(len(traj))]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        print(np.max(gradius))
        return gradius


    def get_durations(self, trajs):
        d = []
        traj_num = 0
        for traj in trajs:
            num = 1
            k = len(traj)
            traj_num += k
            for i, lc in enumerate(traj[1:]):
                if lc == traj[i]:
                    num += 1
                else:
                    d.append(num)
                    num = 1
            if num > 1:
                d.append(num)

        return np.array(d) / self.max_len

    def get_time_durations(self, trajs):
        d = []
        for traj in trajs:
            num = 1
            for i, lc in enumerate(traj[1:]):
                if lc == traj[i]:
                    num += 1
                else:
                    d.append(num)
                    num = 1

        return np.array(d) / self.max_time

    def get_time_durations2(self, trajs):
        d = []
        for traj in trajs:
            for i in range(len(traj) - 1):
                d.append(traj[i + 1] - traj[i])

        return np.array(d)

    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :return:
        """
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj))) / self.max_len)
        reps = np.array(reps, dtype=float)
        return reps

    def get_location_frequency(self, trajs):
        location_cnt = np.zeros(self.max_locs, dtype=np.int32)

        for traj in trajs:
            for rid in traj:
                location_cnt[rid] += 1

        return location_cnt

    def read_road2grid(self):

        road2grid_file = f"./data/{self.data}/road2grid_2.json"
        if not os.path.exists(road2grid_file):
            with open("data/{}/bj_rid_gps.json".format(self.data), "r") as f:
                road_gps = json.load(f)
            road2grid = {}
            for road in road_gps:
                gps = road_gps[road]
                x = math.ceil((gps[0] - self.lon_0) / self.img_unit)
                y = math.ceil((gps[1] - self.lat_0) / self.img_unit)
                road2grid[road] = (x, y)
            with open(road2grid_file, "w") as f:
                json.dump(road2grid, f)
        else:
            with open(road2grid_file, "r") as f:
                road2grid = json.load(f)
        return road2grid

    def get_od_flow(self, trajs):
        total_grid = self.img_width * self.img_height
        grid_od_cnt = np.zeros((total_grid, total_grid), dtype=np.int32)  # 存储网格OD的访问次数
        road2grid = self.read_road2grid()
        for traj in trajs:
            start_rid_grid = road2grid[str(traj[0])]
            des_rid_grid = road2grid[str(traj[-1])]

            start_rid_grid_index = start_rid_grid[0] * self.img_height + start_rid_grid[1]
            des_rid_grid_index = des_rid_grid[0] * self.img_height + des_rid_grid[1]

            grid_od_cnt[start_rid_grid_index][des_rid_grid_index] += 1

        print(np.max(grid_od_cnt))

        return grid_od_cnt.flatten()

    def get_gps_list(self, traj):
        gps_list = []
        for t in traj:
            gps_list.append([self.X[t], self.Y[t]])
        gps_list_array = np.array(gps_list)
        return gps_list_array

    def draw_pict(self, data):
        plt.hist(data, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.show()

    def get_individual_jsds(self, t1, t2, time1, time2):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """


        d1 = self.get_near_distances(t1)
        d2 = self.get_near_distances(t2)
        #
        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance2, 10000)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance2, 10000)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)

        g1 = self.get_radius(t1)
        g2 = self.get_radius(t2)



        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_radius, 1000)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_radius, 1000)


        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)


        od1 = self.get_od_flow(t1)
        od2 = self.get_od_flow(t2)

        od1_dist, _ = EvalUtils.arr_to_distribution(
            od1, 1, 100, 100)
        od2_dist, _ = EvalUtils.arr_to_distribution(
            od2, 1, 100, 100)

        od_jsd = EvalUtils.get_js_divergence(od1_dist, od2_dist)

        # Duration
        # du1 = self.get_durations(t1)
        # # print("------------------")
        # du2 = self.get_durations(t2)
        # du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, self.max_len)
        # du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, self.max_len)
        # du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)

        du1 = self.get_time_durations2(time1)
        # print("------------------")
        du2 = self.get_time_durations2(time2)
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 100, 100)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 100, 100)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)

        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, self.max_len)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, self.max_len)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        # Location Frequency
        l1 = self.get_location_frequency(t1)
        l2 = self.get_location_frequency(t2)
        l_jsd = EvalUtils.get_js_divergence(l1, l2)

        # l1 =  CollectiveEval.get_visits(t1,self.max_locs)
        # l2 =  CollectiveEval.get_visits(t2,self.max_locs)
        # l1_dist, _ = CollectiveEval.get_topk_visits(l1, self.top_k_G)
        # l2_dist, _ = CollectiveEval.get_topk_visits(l2, self.top_k_G)
        # l1_dist2, _ = EvalUtils.arr_to_distribution(l1_dist,0,1,self.top_k_G)
        # l2_dist2, _ = EvalUtils.arr_to_distribution(l2_dist,0,1,self.top_k_G)
        # l_jsd = EvalUtils.get_js_divergence(l1_dist2, l2_dist2)
        #
        # I-rank：计算每个轨迹中访问最频繁的前k个位置及其对应的频率
        f1 = self.get_overall_topk_visits_freq(t1, self.top_k_I)
        f2 = self.get_overall_topk_visits_freq(t2, self.top_k_I)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1, 0, 1, self.top_k_I)
        f2_dist, _ = EvalUtils.arr_to_distribution(f2, 0, 1, self.top_k_I)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)

        return d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, od_jsd



    def min_micro_distance3(self, gen_list, real_test, k):
        orgin_gen = gen_list[0]

        count = 0
        history_sspd_distance = [0]
        history_lcss_distance_GPS = [0]
        history_frechet_distance = []

        generate_gps_list = self.get_gps_list(gen_list)
        generate_gps_list = np.array(generate_gps_list)
        k = 0
        for real_list in real_test:
            orgin_test = real_list[0]
            # des_test=real_list[-1]
            if orgin_gen == orgin_test:
                k += 1
                trace = self.get_gps_list(real_list)
                trace_gps_list = np.array(trace)


                frechet = Frechet_distance(trace_gps_list, generate_gps_list)
                #

                history_frechet_distance.append(frechet)

        if k == 0:
            for real_list in real_test:
                trace = self.get_gps_list(real_list)
                trace_gps_list = np.array(trace)

                frechet = Frechet_distance(trace_gps_list, generate_gps_list)
                #
                history_frechet_distance.append(frechet)

        min_sspd = np.min(history_sspd_distance)
        min_lcss = np.min(history_lcss_distance_GPS)
        min_frechet = np.min(history_frechet_distance)

        return min_sspd, min_lcss, min_frechet, count



    def get_point_query_error(self, t1, t2):

        query_num = 500
        size_factor = 9

        queries = [
            SquareQuery(self.lon_0, self.lat_0, self.lon_1, self.lat_1, size_factor=size_factor)
            for _ in range(query_num)]

        real = []
        for traj in t1:
            t = []
            for rid in traj:
                point = tuple(self.road_gps[str(rid)])
                t.append(point)
            real.append(t)

        gen = []
        for traj in t2:
            t = []
            for rid in traj:
                point = tuple(self.road_gps[str(rid)])
                t.append(point)
            gen.append(t)

        query_error = experiment.calculate_point_query(real,
                                                       gen,
                                                       queries)

        return query_error

    def get_road_query_error(self, t1, t2):

        query_num = self.query_num

        loaded_numbers = np.random.choice(10904, query_num, replace=False)

        queries = [
            RoadQuery(loaded_numbers[i])
            for i in range(query_num)]

        query_error = experiment.calculate_road_query(t1,
                                                      t2,
                                                      queries)

        return query_error

    def get_diameter(self, t):
        max_d = 0
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                max_d = max(max_d, euclidean(t[i], t[j]))

        return max_d

    def get_diameter_error(self, orig_db,
                           syn_db,
                           bucket_num=20):

        orig_diameter = [self.get_diameter(t) for t in tqdm(orig_db)]
        syn_diameter = [self.get_diameter(t) for t in tqdm(syn_db)]

        d1_dist, _ = EvalUtils.arr_to_distribution(
            orig_diameter, min(orig_diameter), max(orig_diameter), bucket_num)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            syn_diameter, min(orig_diameter), max(orig_diameter), bucket_num)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)

        return d_jsd

    def get_hausdorff_metric(self, t1, t2):

        length = len(t2)
        total_edit_distance_GPS = 0
        total_edit_distance_ID = 0
        total_dtw = 0
        total_hausdorff = 0
        total_frechet = 0
        total_sspd = 0
        total_lcss = 0
        count = 0
        for i in tqdm(range(length), desc="Calculate Micro"):
            min_dtw, min_hasudorf, min_edr_GPS,min_edr_ID,count_no_od=self.min_micro_distance2(t2[i],t1,i)

            total_hausdorff+=min_hasudorf


        return total_edit_distance_GPS/(length),total_edit_distance_ID/(length),total_dtw/(length),total_hausdorff/(length)

    def get_micro_metric(self, t1, t2):

        length = len(t2)
        total_edit_distance_GPS = 0
        total_edit_distance_ID = 0
        total_dtw = 0
        total_hausdorff = 0
        total_frechet = 0
        total_sspd = 0
        total_lcss = 0
        count = 0
        for i in tqdm(range(length), desc="Calculate Micro"):
            min_sspd, min_lcss, min_frechet, count_no_od = self.min_micro_distance3(t2[i], t1, i)
            total_sspd += min_sspd
            total_lcss += min_lcss
            total_frechet += min_frechet
            count += count_no_od

        return total_sspd / (length), total_lcss / (length), total_frechet / (length)

    def get_bleu_metric(self, gen_list, real_test, k):
        orgin_gen = gen_list[0]
        count = 0

        real_trajs = []

        s = 0
        for real_list in real_test:
            orgin_test = real_list[0]
            # des_test=real_list[-1]
            if orgin_gen == orgin_test:
                s += 1
                real_trajs.append(real_list)

        if s == 1 or s == 0:
            s = 0

            real_trajs = []
            real_trajs.extend(real_test)

        bleu_score = sentence_bleu(real_trajs, gen_list, smoothing_function=smoothie)

        return bleu_score

    # 将int类型的路段ID序列转化为str

    def int2str(self, trajs):

        output = []
        for traj in trajs:
            str_list = [str(item) for item in traj]
            output.append(str_list)

        return output


    def get_error_metric(self, t1, t2):
        real = []
        for traj in t1:
            t = []
            for rid in traj:
                point = tuple(self.road_gps[str(rid)])
                t.append(point)
            real.append(t)

        gen = []
        for traj in t2:
            t = []
            for rid in traj:
                point = tuple(self.road_gps[str(rid)])
                t.append(point)
            gen.append(t)

        query_error = self.get_road_query_error(t1, t2)

        diameter_error =0

        return query_error, diameter_error


    def get_semantic_metric(self, t1, t2):



        road2grid = self.read_road2grid()

        orig_grid_trajectories = []
        for traj in t1:
            t = []
            for rid in traj:
                rid_grid = road2grid[str(rid)]
                rid_grid_index = rid_grid[0] * self.img_height + rid_grid[1]
                t.append(rid_grid_index)
            orig_grid_trajectories.append(t)

        synthetic_grid_trajectories = []
        for traj in t2:
            t = []
            for rid in traj:
                rid_grid = road2grid[str(rid)]
                rid_grid_index = rid_grid[0] * self.img_height + rid_grid[1]
                t.append(rid_grid_index)
            synthetic_grid_trajectories.append(t)

        orig_pattern = experiment.mine_patterns(orig_grid_trajectories)
        syn_pattern = experiment.mine_patterns(synthetic_grid_trajectories)

        pattern_f1_error = experiment.calculate_pattern_f1_error(orig_pattern, syn_pattern,self.pattern_num)
        pattern_support_error = experiment.calculate_pattern_support(orig_pattern, syn_pattern,self.pattern_num)

        return pattern_f1_error, pattern_support_error




class CollectiveEval(object):
    """
    collective evaluation metrics
    """

    @staticmethod
    def get_visits(trajs, max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float)
        for traj in trajs:
            for t in traj:
                visits[t] += 1
        visits = visits / np.sum(visits)
        return visits

    @staticmethod
    def get_topk_visits(visits, K):  # 计算前100名访问频率最高的地点的访问频率
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs

    @staticmethod
    def get_topk_accuracy(v1, v2, K):
        """
        get the accuracy of top-k visiting locations
        :param v1:
        :param v2:
        :param K:
        :return:
        """
        _, tl1 = CollectiveEval.get_topk_visits(v1, K)
        _, tl2 = CollectiveEval.get_topk_visits(v2, K)
        coml = set(tl1) & set(tl2)
        return len(coml) / K


def evaluate(datasets, model_name):
    individualEval = IndividualEval(data=datasets)
    logger = get_logger(name='eval {}-{}'.format(datasets, model_name))
    test_data = read_data_from_file2('./data/%s/test.data' % opt.datasets)
    gen_data = read_data_from_file2('./data/%s/gen.data' % opt.datasets)
    test_time = read_data_from_file2('./data/%s/test_time.data' % opt.datasets)
    gen_time = read_data_from_file2('./data/%s/gen_time.data' % opt.datasets)
    d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, od_jsd = individualEval.get_individual_jsds(test_data, gen_data,
                                                                                           test_time, gen_time)

    logger.info('evaluate traj:{}'.format(len(gen_data)))
    logger.info('Distance:{}, Radius:{}, Duration:{}'.format(d_jsd, g_jsd, du_jsd))
    logger.info('OD_flow:{}'.format(od_jsd))
    logger.info('DailyLoc:{},Location Frequency:{}, I-rank:{},'.format(p_jsd, l_jsd, f_jsd))

    query_error, diameter_error = individualEval.get_error_metric(test_data, gen_data)
    logger.info('query_error:{},diameter_error:{}'.format(query_error, diameter_error))


    Pattern_score, Pattern_Error = individualEval.get_semantic_metric(test_data, gen_data)
    logger.info('Pattern_score:{},Pattern_Error:{}'.format(Pattern_score, Pattern_Error))
    #
    #bleu 和geo_bleu
    #
    # bleu = individualEval.get_micro_metric2(test_data, gen_data)

    # logger.info('bleu:{}'.format(bleu))
    #
    haus=individualEval.get_hausdorff_metric(test_data,gen_data)
    logger.info('haus:{}'.format(haus))

    sspd,lcss,fre=individualEval.get_micro_metric(test_data,gen_data)
    logger.info('sspd:{},lcss:{},frechet:{}'.format(sspd,lcss,fre))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--datasets', default='BJ_Taxi', type=str,
                        choices=["BJ_Taxi", "Porto_Taxi"])
    parser.add_argument('--model_name', default='KG-TrajGen', type=str, choices=["TS-TrajGAN", "MoveSim", "STEGA"])
    opt = parser.parse_args()

    if opt.datasets == 'Porto_Taxi':
        max_locs = 10904
    elif opt.datasets == "BJ_Taxi":
        max_locs = 37684


    evaluate(opt.datasets, opt.model_name)
