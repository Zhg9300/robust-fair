import random
import time
import traceback
import numpy as np
import torch

from ByrdLab import FEATURE_TYPE

def log(*k, **kw):
    timeStamp = time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)
    # sys.stdout.flush()

def project(y):
    ''' algorithm comes from:
    https://arxiv.org/pdf/1309.1541.pdf
    '''
    u = sorted(y, reverse=True)
    x = []
    rho = 0
    for i in range(len(y)):
        if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i]))) > 0:
            rho = i + 1
    lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
    for i in range(len(y)):
        x.append(max(y[i]+lambda_, 0))
    return x

def NBS(grads_bf_comm, hs, beta):
    '''norm-base screening: input: local gradients, screening percentage'''
    norm_of_grads = [torch.norm(grad) for grad in grads_bf_comm]
    sorted_indices = sorted(range(len(norm_of_grads)), key=norm_of_grads.__getitem__, reverse=True)
    #print(sorted_indices)
    num_to_remove = int(beta * len(grads_bf_comm))
    new_grads_bf_comm = [grads_bf_comm[i] for i in range(len(grads_bf_comm)) if
                         i not in sorted_indices[:num_to_remove]]
    new_grads_bf_comm = torch.stack(new_grads_bf_comm)
    new_hs = [hs[i] for i in range(len(grads_bf_comm)) if
                         i not in sorted_indices[:num_to_remove]]

    denominator = sum(new_hs)
    agg_res = torch.mean(new_grads_bf_comm, dim=0)
    return denominator, agg_res

def NBS_AFL(grads_bf_comm, beta):
    '''norm-base screening: input: local gradients, screening percentage'''
    norm_of_grads = [torch.norm(grad) for grad in grads_bf_comm]
    sorted_indices = sorted(range(len(norm_of_grads)), key=norm_of_grads.__getitem__, reverse=True)
    #print(sorted_indices)
    num_to_remove = int(beta * len(grads_bf_comm))
    new_grads_bf_comm = [grads_bf_comm[i] for i in range(len(grads_bf_comm)) if
                         i not in sorted_indices[:num_to_remove]]
    new_grads_bf_comm = torch.stack(new_grads_bf_comm)
    agg_res = torch.mean(new_grads_bf_comm, dim=0)
    return agg_res

def adapt_model_type(model):
    if FEATURE_TYPE == torch.float64:
        return model.double()
    elif FEATURE_TYPE == torch.float32:
        return model.float()
    elif FEATURE_TYPE == torch.float16:
        return model.half()

# function decorator: fix seed
# TODO: the 'construc_rng_pack' function in class 'distributedOptimizer'
#      should consider GPU and cpu generators
# Warning: having used in the code. Develops are recommanded to use the RngPack
#          given in ByrdLab.library.RandomNumberGenerator to control the random
#          seed
def fix_seed(run):
    def wrapper(self, *args, **kw):
        # fit seed
        if self.fix_seed:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                self._cuda_deterministic = torch.backends.cudnn.deterministic
                torch.backends.cudnn.deterministic = True
        result = run(self, *args, **kw)
        # reset the random seed
        if self.fix_seed:
            seed = time.time()
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = self._cuda_deterministic
        return result        
    return wrapper


# function decorator: program won't stop even when exception raise
def no_exception_blocking(func):
    def wrapper(*args, **kw):
        try:
            return func(*args, **kw)
        except Exception as e:
            traceback.print_exc()
    return wrapper

def get_model_param(model, use_str=True):
    para = sum([x.nelement() for x in model.parameters()])
    if not use_str:
        return para
    elif para >= 1 << 30:
        return '{:.2f}G'.format(para / (1<<30))
    elif para >= 1 << 20:
        return '{:.2f}M'.format(para / (1<<20))
    elif para >= 1 << 10:
        return '{:.2f}K'.format(para / (1<<10))
    else:
        return str(para)
    
            
def MH_rule(graph):
    # Metropolis-Hastings rule
    node_size = graph.number_of_nodes()
    W = torch.eye(node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(node_size):
            if i == j or not graph.has_edge(j, i):
                continue
            i_n = graph.neighbor_sizes[i] + 1
            j_n = graph.neighbor_sizes[j] + 1
            W[i][j] = 1 / max(i_n, j_n)
            W[i][i] -= W[i][j]
    return W
    
def naive_local_avg(graph):
    node_size = graph.number_of_nodes()
    W = torch.zeros((node_size, node_size), dtype=FEATURE_TYPE)
    for i in range(node_size):
        neigbor_size = graph.neighbor_sizes[i] + 1
        for j in range(node_size):
            if i != j and not graph.has_edge(j, i):
                continue
            W[i][j] = 1 / neigbor_size
    return W
    