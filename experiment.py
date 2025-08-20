import random
from typing import List, Tuple
import utils
import numpy as np
import multiprocessing
import math
from tqdm import tqdm

CORES = multiprocessing.cpu_count() // 2


class Query:
    def __init__(self):
        pass

    def point_query(self, db):
        raise NotImplementedError

class SquareQuery(Query):
    def __init__(self,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float,
                 size_factor=9.0):
        super().__init__()
        # Randomly select center
        center_x = random.random() * (max_x - min_x) + min_x
        center_y = random.random() * (max_y - min_y) + min_y
        self.center = (center_x, center_y)

        self.edge = math.sqrt((max_x-min_x)*(max_y-min_y)/size_factor)
        self.left_x = center_x - self.edge / 2
        self.up_y = center_y + self.edge / 2
        self.right_x = center_x + self.edge / 2
        self.down_y = center_y - self.edge / 2

    def in_square(self, point: Tuple[float, float]):
        return self.left_x <= point[0] <= self.right_x and self.down_y <= point[1] <= self.up_y


    def point_query(self, db: List[List[Tuple[float, float]]]):
        count = 0
        for t in db:
            for p in t:
                if self.in_square(p):
                    count += 1

        return count


class RoadQuery(Query):
    def __init__(self,
                 road_num
                 ):
        super().__init__()
        self.road_ID = road_num

    def point_query(self, db: List[List[Tuple[float, float]]]):
        count = 0
        for t in db:
            for p in t:
                if p==self.road_ID:  #
                    count += 1
                    break

        return count



class Pattern:
    def __init__(self, grids):
        self.grids = grids

    @property
    def size(self):
        return len(self.grids)

    def __eq__(self, other):
        if other is None:
            return False
        if not type(other) == Pattern:
            return False
        if not other.size == self.size:
            return False
        for i in range(self.size):
            if not self.grids[i] == other.grids[i]:
                return False

        return True

    def __hash__(self):
        prime = 31
        result = 1
        for g in self.grids:
            result = result * prime + g.__hash__()

        return result


def calculate_point_query(orig_db,
                          syn_db,
                          queries: List[Query],
                          sanity_bound=0.01):
    actual_ans = list()
    syn_ans = list()

    total_points = np.sum([len(t) for t in orig_db])

    for q in tqdm(queries,total=len(queries)):
        actual_ans.append(q.point_query(orig_db))
        syn_ans.append(q.point_query(syn_db))

    actual_ans = np.asarray(actual_ans)
    syn_ans = np.asarray(syn_ans)

    numerator = np.abs(actual_ans - syn_ans)
    denominator = np.asarray([max(actual_ans[i], total_points * sanity_bound) for i in range(len(actual_ans))])

    error = numerator / denominator

    return np.mean(error)




def mine_patterns(db, min_size=2, max_size=8):
    """
    Find all patterns of size between min_size and max_size
    :return: Dict[Pattern, int]: count of each pattern
    """
    pattern_dict = {}
    for curr_size in range(min_size, max_size + 1):
        for t in db:
            for i in range(0, len(t) - curr_size + 1):
                p = Pattern(t[i: i + curr_size])
                try:
                    pattern_dict[p] += 1
                except KeyError:
                    pattern_dict[p] = 1

    return pattern_dict
#
#
def calculate_pattern_f1_error(orig_pattern,syn_pattern,k):   #500  porto300
    sorted_orig = sorted(orig_pattern.items(), key=lambda x: x[1], reverse=True)
    sorted_syn = sorted(syn_pattern.items(), key=lambda x: x[1], reverse=True)

    orig_top_k = [x[0] for x in sorted_orig][:k]
    syn_top_k = [x[0] for x in sorted_syn][:k]

    count = 0
    for p1 in syn_top_k:
        if p1 in orig_top_k:
            count += 1

    precision = count / k
    recall = count / k

    return 2 * precision * recall / (precision + recall)
#

#
def calculate_pattern_support(orig_pattern, syn_pattern,k):
    sorted_orig = sorted(orig_pattern.items(), key=lambda x: x[1], reverse=True)
    orig_top_k = [x[0] for x in sorted_orig][:k]

    error = 0
    for i in tqdm(range(len(orig_top_k))):
        p: Pattern = orig_top_k[i]
        orig_support = orig_pattern[p]
        try:
            syn_support = syn_pattern[p]
        except KeyError:
            syn_support = 0
        error += np.abs(orig_support-syn_support)/orig_support

    return error / k
