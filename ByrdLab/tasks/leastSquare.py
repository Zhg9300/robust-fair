from ByrdLab.library.dataset import DataPackage, StackedDataSet
from ByrdLab import FEATURE_TYPE, VALUE_TYPE
from ByrdLab.library.RandomNumberGenerator import RngPackage, torch_rng
from functools import partial

import torch

from ByrdLab.tasks import Task
from ByrdLab.library.initialize import RandomInitialize

class LeastSqaure_LinearModel(torch.nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=feature_dimension,
                                      out_features=1, bias=True)
    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.linear(features)

def train_full_batch_generator(dist_dataset, node,
                               rng_pack: RngPackage=RngPackage()):
    while True:
        yield dist_dataset[node][:]
        
# def test_full_batch_generator(dataset):
#     yield dataset[:]
def test_full_batch_generator(dist_dataset, node,
                               rng_pack: RngPackage=RngPackage()):
    while True:
        yield dist_dataset[node][:]

def least_square_loss(predictions, targets):
    return ((predictions-targets)**2).mean() / 2 #原来没有/2

class LeastSquareToySet(DataPackage):
    def __init__(self, set_size, dimension, noise=0.1, w_star=None,
                 fix_seed=False, seed=1):
        generator = torch_rng(seed) if fix_seed else None
        if w_star == None:
            w_1 = torch.normal(-1, 1, (dimension, 1), dtype=FEATURE_TYPE,
                                   generator=generator)
            print('w1  ', w_1.shape)
            w_2 = torch.normal(1, 1, (dimension, 1), dtype=FEATURE_TYPE,
                               generator=generator)
            print('w2  ', w_2.shape)
            # w_star = 1*torch.randn(dimension, dtype=FEATURE_TYPE,
            #                        generator=generator)
        #assert w_star.size() == torch.Size([dimension])
        #assert w_1.size() == torch.Size([dimension])
        #self.w_star = w_star
        self.noise = noise
        # X = torch.randn((set_size, dimension), dtype=FEATURE_TYPE,
        #                 generator=generator)
        # X.add_(torch.rand(1, generator=generator))
        X1 = torch.normal(-2, 2, (int(0.1 * set_size), dimension), dtype=FEATURE_TYPE,
                                         generator=generator)
        #print('X!  ', X1.shape)
        X2 = torch.normal(2, 2, (int(0.9 * set_size), dimension), dtype=FEATURE_TYPE,
                         generator=generator)
        #X1.add_(torch.rand(1, generator=generator))
        #X2.add_(torch.rand(1, generator=generator))
        X = torch.cat([X1, X2], dim=0)
        Y1 = torch.matmul(X1, w_1)
        #print('Y1', Y1)
        Y2 = torch.matmul(X2, w_2)
        print('Y2', Y2)
        Y = torch.cat([Y1, Y2])
        # X.add_(torch.rand(dimension))
        # for i in range(set_size):
        #     X[i].div_(X[i].norm())
        #Y = torch.matmul(X, w_star)
        Y.add_(torch.randn(Y.shape, dtype=VALUE_TYPE, generator=generator),
               alpha=noise)
        name = f'ToySet_D={dimension}_N={set_size}'
        dataset = StackedDataSet(features=X, targets=Y)
        super().__init__(name=name, train_set=dataset, test_set=dataset)


def random_generator(dataset, batch_size=1, rng_pack: RngPackage=RngPackage()):
    while True:
        beg = rng_pack.random.randint(0, len(dataset)-1)
        if beg+batch_size <= len(dataset):
            yield dataset[beg:beg+batch_size]
        else:
            features, targets = zip(dataset[beg:beg+batch_size],
                                    dataset[0:(beg+batch_size) % len(dataset)])
            yield torch.cat(features), torch.cat(targets)
def order_generator(dataset, batch_size=1, rng_pack: RngPackage=RngPackage()):
    beg = 0
    while beg < len(dataset):
        end = min(beg+batch_size, len(dataset))
        yield dataset[beg:end]
        beg += batch_size

class LeastSquareToyTask(Task):
    def __init__(self):
        #model = None

        # dimension = 100
        # data_cnt = 10
        dimension = 100
        data_cnt = 5000 #50
        data_package = LeastSquareToySet(data_cnt, dimension,
                                         # noise=0,
                                         fix_seed=True)
        model = LeastSqaure_LinearModel(data_package.feature_dimension)
        super_params = {
            'rounds': 20,#20
            'display_interval': 1000,#1000
            # 'primal_weight_decay': 5e-3,
            # 'dual_weight_decay': 1e-3,
            'penalty': 6e-6,#6e-2
            # 'penalty': 1.37e-1, # minimum penalty
            'lr': 1e-3, #3e-2
        }
        #get_train_iter = train_full_batch_generator
        test_set = data_package.test_set
        get_train_iter = partial(random_generator,
                                 batch_size=64)#32
        get_test_iter = partial(order_generator, dataset=test_set, batch_size=1000)
        # get_test_iter = partial(test_full_batch_generator,
        #                         dataset=data_package.test_set)
        loss_fn = least_square_loss
        super().__init__(weight_decay=0.05, data_package=data_package, model=model,
                         loss_fn=loss_fn,
                         test_fn=loss_fn, #None
                         get_train_iter=get_train_iter,
                         get_test_iter=get_test_iter,
                         initialize_fn=RandomInitialize(),
                         super_params=super_params,
                         name=f'LS_{data_package.name}', model_name='LinearModel')


