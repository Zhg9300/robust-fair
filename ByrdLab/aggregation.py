import copy
import itertools
import math

import torch
from ByrdLab import FEATURE_TYPE
from scipy import stats

from ByrdLab.library.tool import MH_rule

def mean(wList):
    return torch.mean(wList, dim=0)

def weight_mean(wList, weight):
    return torch.tensordot(torch.tensor(weight, dtype=FEATURE_TYPE), wList, dims=1)


def geometric_median(wList, max_iter=80, err=1e-5):#max_iter=80
    guess = median(wList)
    #guess = torch.mean(wList, dim=0)
    #guess = wList[0]
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack(
            [w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= err:
            break
    return guess


def medoid_index(wList):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    dist_sum = dist.sum(dim=1)
    return dist_sum.argmax()


def medoid(wList):
    return wList[medoid_index(wList)]


def Krum_index(wList, byzantine_size):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()**2
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    # The distance from any node to itself must be 0.00, so we add 1 here
    k = node_size - byzantine_size - 2 + 1
    topv, _ = dist.topk(k=k, dim=1)
    scores = topv.sum(dim=1)
    return scores.argmax()


def Krum(wList, byzantine_size):
    index = Krum_index(wList, byzantine_size)
    return wList[index]


def mKrum(wList, byzantine_size, m=2):
    remain = wList
    result = torch.zeros_like(wList[0], dtype=FEATURE_TYPE)
    for _ in range(m):
        res_index = Krum_index(remain, byzantine_size)
        result += remain[res_index]
        remain = remain[torch.arange(remain.size(0)) != res_index]
    return result / m


def median(wList):
    return wList.median(dim=0)[0]


def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
      data Indexable (including ability to query length) containing the elements
    Returns:
      Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (data[i], data[j])


def brute_selection(gradients, f, **kwargs):
    """ Brute rule. 
    brute is also called minimum diameter averaging (MDA)
    The code comes from:
    https://github.com/LPD-EPFL/Garfield/blob/master/pytorch_impl/libs/aggregators/brute.py#L32

    Args:
      gradients Non-empty list of gradients to aggregate
      f         Number of Byzantine gradients to tolerate
      ...       Ignored keyword-arguments
    Returns:
      Selection index set
    """
    n = len(gradients)
    # Compute all pairwise distances
    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
        distances[i] = gradients[x].sub(gradients[y]).norm().item()
    # Select the set of smallest diameter
    sel_iset = None
    sel_diam = None
    for cur_iset in itertools.combinations(range(n), n - f):
        # Compute the current diameter (max of pairwise distances)
        cur_diam = 0.
        for x, y in pairwise(cur_iset):
            # Get distance between these two gradients ("magic" formula valid since x < y)
            cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
            # Check finite distance (non-Byzantine gradient must only contain finite coordinates), drop set if non-finite
            if not math.isfinite(cur_dist):
                break
            # Check if new maximum
            if cur_dist > cur_diam:
                cur_diam = cur_dist
        else:
            # Check if new selected diameter
            if sel_iset is None or cur_diam < sel_diam:
                sel_iset = cur_iset
                sel_diam = cur_diam
    # Return the selected gradients
    assert sel_iset is not None, "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
    return sel_iset


def brute(gradients, byzantine_size, **kwargs):
    """ Brute rule.
    Args:
      gradients Non-empty list of gradients to aggregate
      f         Number of Byzantine gradients to tolerate
      ...       Ignored keyword-arguments
    Returns:
      Aggregated gradient
    """
    sel_iset = brute_selection(gradients, byzantine_size, **kwargs)
    return sum(gradients[i] for i in sel_iset).div_(len(gradients) - byzantine_size)


def trimmed_mean(wList, byzantine_size):
    node_size = wList.size(0)
    proportion_to_cut = byzantine_size / node_size
    #tm_np = stats.trim_mean(wList, proportion_to_cut, axis=0)
    tm_np = stats.trim_mean(wList.detach().numpy(), proportion_to_cut, axis=0)
    return torch.from_numpy(tm_np)


def remove_outliers(wList, byzantine_size):
    mean = torch.mean(wList, dim=0)
    # remove the largest 'byzantine_size' model
    distances = torch.tensor([
        -torch.norm(model - mean) for model in wList
    ])
    node_size = wList.size(0)
    remain_cnt = node_size - byzantine_size
    (_, remove_index) = torch.topk(distances, k=remain_cnt)
    return wList[remove_index].mean(dim=0)


def faba(wList, byzantine_size):
    remain = wList
    for _ in range(byzantine_size):
        mean = remain.mean(dim=0)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
            torch.norm(model - mean) for model in remain
        ])
        remove_index = distances.argmax()
        remain = remain[torch.arange(remain.size(0)) != remove_index]
    return remain.mean(dim=0)


def bulyan(wList, byzantine_size):
    remain = wList
    selected_ls = []
    node_size = wList.size(0)
    selection_size = node_size-2*byzantine_size
    for _ in range(selection_size):
        res_index = Krum_index(remain, byzantine_size)
        selected_ls.append(remain[res_index])
        remain = remain[torch.arange(remain.size(0)) != res_index]
    selection = torch.stack(selected_ls)
    m = median(selection)
    dist = -(selection - m).abs()
    indices = dist.topk(k=selection_size-2*byzantine_size, dim=0)[1]
    if len(wList.size()) == 1:
        result = selection[indices].mean()
    else:
        result = torch.stack([
            selection[indices[:, d], d].mean() for d in range(wList.size(1))])
    return result

class DRO_meanW(DecentralizedAggregation):
    def __init__(self, graph):
        super(DRO_meanW, self).__init__(name='DRO_meanW', graph=graph)
    def run(self, local_models, node, weight_vec):
        return torch.tensordot(torch.tensor(weight_vec), local_models, dims=1)

class qFedAvg(DecentralizedAggregation):
    def __init__(self, graph):
        super(qFedAvg, self).__init__(name='qFedAvg', graph=graph)
    def run(self, weights, Deltas, hs):
        demominator = torch.sum(torch.tensor(hs))
        updates = torch.sum(torch.sum(torch.stack(Deltas)) * 1.0 / demominator)
        new_solutions = weights - updates
        return new_solutions

class centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold=10):
        super().__init__(name=f'CC_tau={threshold}', graph=graph)
        self.memory = None
        self.threshold = threshold
        self.central = None
    def run(self, local_models):
        if self.memory == None:
            self.memory = torch.zeros_like(local_models[0])
            self.central = torch.zeros_like(local_models[0])
        diff = torch.zeros_like(local_models[0])
        for n in range(10):
            model = local_models[n]
            norm = (model - self.memory).norm()
            if norm > self.threshold:
                diff += self.threshold * (model - self.memory) / norm
            else:
                diff += model - self.memory
        diff /= 10
        self.memory = self.memory + diff
        return self.memory
                cum_diff += weight * diff
        return local_model + cum_diff
