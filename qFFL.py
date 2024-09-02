import argparse
import datetime
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from ByrdLab.aggregation import (qFedAvg, mean,
                                 Krum, mKrum, trimmed_mean, geometric_median, median, faba, bulyan, brute)
from ByrdLab.attack import (sign_flipping, gaussian, disguise)
from ByrdLab.decentralizedAlgorithm import DSGD, CSGD, AFL, qFFL
from ByrdLab.graph import CompleteGraph
from ByrdLab.library.cache_io import dump_file_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                       iidPartition)
from ByrdLab.library.tool import log, NBS
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask
import torch

parser = argparse.ArgumentParser(description='Robust Temporal Difference Learning')

# Arguments
parser.add_argument('--graph', type=str, default='CompleteGraph')
parser.add_argument('--aggregation', type=str,
                    default='mean')  # mean, trimmed-mean, median, geometric-median, mKrum, Krum, bulyan, cc, scc
parser.add_argument('--attack', type=str,
                    default='disguise')  # disguise, sign_flipping, gaussian, sample_duplicate, zero_sum, zero_value, alie
parser.add_argument('--data-partition', type=str, default='noniid')  # default iid
parser.add_argument('--lr-ctrl', type=str, default='1/sqrt k')

parser.add_argument('--no-fixed-seed', action='store_true',
                    help="If specifed, the random seed won't be fixed")
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--without-record', action='store_true',
                    help='If specifed, no file of running record and log will be left')

args = parser.parse_args()

# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
if args.graph == 'CompleteGraph':
    graph = CompleteGraph(node_size=10, byzantine_size=2)
else:
    assert False, 'unknown graph'

if args.attack == 'none':
    graph = graph.honest_subgraph()
# ===========================================

# -------------------------------------------
# define learning task

data_package = fashionmnist()
#task = softmaxRegressionTask(data_package)
task = NeuralNetworkTask(data_package)
# ===========================================


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
    # super_params = task.super_params
    # total_iterations = super_params['rounds']*super_params['display_interval']
    # lr_ctrl = one_over_sqrt_k_lr(total_iteration=total_iterations,
    #                              a=math.sqrt(1001), b=1000)
elif args.lr_ctrl == 'ladder':
    decreasing_iter_ls = [30000, 60000]
    proportion_ls = [0.5, 0.2]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'
# ===========================================


# -------------------------------------------
# define data partition
# -------------------------------------------
if args.data_partition == 'trival':
    partition_cls = TrivalPartition
elif args.data_partition == 'iid':
    partition_cls = iidPartition
elif args.data_partition == 'noniid':
    partition_cls = LabelSeperation
else:
    assert False, 'unknown data-partition'
# ===========================================


# -------------------------------------------
# define aggregation
# -------------------------------------------
if args.aggregation == 'mean':
    aggregation = mean
elif args.aggregation == 'NBS':
    aggregation = NBS
elif args.aggregation == 'median':
    aggregation = median
elif args.aggregation == 'trimmed-mean':
    aggregation = trimmed_mean
elif args.aggregation == 'geometric-median':
    aggregation = geometric_median
elif args.aggregation == 'Krum':
    aggregation = Krum
elif args.aggregation == 'mKrum':
    aggregation = mKrum
elif args.aggregation == 'brute':
    aggregation = brute
elif args.aggregation == 'bulyan':
    aggregation = bulyan
elif args.aggregation == 'faba':
    aggregation = faba
else:
    assert False, 'unknown aggregation'
# ===========================================

# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'disguise':
    attack = disguise
elif args.attack == 'sign_flipping':
    attack = sign_flipping
elif args.attack == 'gaussian':
    attack = gaussian
else:
    assert False, 'unknown attack'

if args.attack == 'none':
    attack_name = 'baseline'
else:
    attack_name = attack
# ===========================================

workspace = []
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record

# initilize optimizer
env = qFFL(aggregation=aggregation, graph=graph, attack=attack,
          weight_decay=task.weight_decay, data_package=task.data_package,
          model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
          initialize_fn=task.initialize_fn,
          get_train_iter=task.get_train_iter,
          get_test_iter=task.get_test_iter,
          partition_cls=partition_cls, lr_ctrl=lr_ctrl,
          fix_seed=fix_seed, seed=seed,
          **task.super_params)

title = '{}_{}_{}'.format(env.name, attack_name, aggregation)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title

data_package = task.data_package
super_params = task.super_params

# print the running information
print('=========================================================')
print(torch.cuda.is_available())
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', data_package.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, aggregation, attack_name))
print('{:12s} lr={} lr_ctrl={}, weight_decay={}'.format(
    '[Optimizer]', super_params['lr'], env.lr_ctrl.name, task.weight_decay))
print('{:12s} graph={}, honest_size={}, byzantine_size={}'.format(
    '[Graph]', graph.name, graph.honest_size, graph.byzantine_size))
print('{:12s} rounds={}, display_interval={}, total iterations={}'.format(
    '[Running]', env.rounds, env.display_interval, env.total_iterations))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', seed, fix_seed))
print('{:12s} record_in_file={}'.format('[System]', record_in_file))
print('-------------------------------------------')

log('[Start Running]')
_, loss_path, acc_path, worst_acc_path, var_path = env.run()

print('-------------------------------------------')

