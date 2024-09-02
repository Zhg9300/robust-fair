import datetime
import numpy as np
import torch
import csv
from ByrdLab.environment import Dec_Byz_Iter_Env, Dec_Byz_Opt_Env
from ByrdLab.library.dataset import EmptySet
from ByrdLab.library.partition import EmptyPartition
from ByrdLab.library.measurements import avg_loss_accuracy_dist, consensus_error, worst_loss_accuracy_dist, worst_loss_dist
from ByrdLab.library.tool import log, project, NBS
from ByrdLab.aggregation import mean, median, Krum
import os
import random

#new
class vanillaCSGD(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(name='CSGD', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation

    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)

        # initial record
        loss_path = []
        acc_path = []
        worst_acc_path = []
        var_path = []
        flag = 0 # 记录是否线性，线性为1
        if type(self.model).__name__ == 'LeastSqaure_LinearModel':
            flag = 1

        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[vanillaCSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, avg acc={:.4f}, worst acc={:.4f}, var={:.4e}, lr={:f}'

        # local models before communication
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack)#if node in self.honest_nodes else None for node in self.nodes]
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            test_iters = [self.get_test_iter(dataset=self.dist_test_set[node], rng_pack=self.rng_pack)#if node in self.honest_nodes else None for node in self.nodes]
                          for node in self.nodes]
            #lr = 0.5
            lr = 0.1
            #lr = self.lr_ctrl.get_lr(iteration)
            # lastest learning rate
            # Collect gradients from honest nodes
            for node in self.nodes: # local gradient descent
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                dist_models.loss_vec[node] = loss  # 更新当前轮次的loss
                model.zero_grad()
                loss.backward()
                cumulated_param = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.nelement()
                        beg, end = cumulated_param, cumulated_param + param.nelement()
                        cumulated_param += param_size
                        dist_models.grads_vec[node][beg:end] = param.grad.view(-1)
                    dist_models.grads_vec[node].view_as(dist_models.params_vec[node])
            grads_bf_comm = dist_models.grads_vec.clone()
            # communication and attack
            if self.attack != None:
                if self.attack.__name__ == 'disguise':
                    for node in self.byzantine_nodes:
                        # dist_models.loss_vec[node] = max([loss_last_round[i] for i in self.honest_nodes])
                        # tar = random.choice([0,1,2,3,5,6,7])
                        grads_bf_comm[node] = grads_bf_comm[4].clone()
                elif self.attack.__name__ == 'sign_flipping':
                     self.attack(grads_bf_comm,self.honest_nodes, self.byzantine_nodes, 20, 0)
                elif self.attack.__name__ == 'gaussian':
                    self.attack(grads_bf_comm, self.honest_nodes, self.byzantine_nodes, 20)

            # aggregate
            beta = 0 if self.attack == None else 0.2

            # aggregation
            if self.aggregation.__name__ == 'mean':
                aggregation_res = self.aggregation(grads_bf_comm)
            elif self.aggregation.__name__ == 'NBS_AFL':
                aggregation_res = self.aggregation(grads_bf_comm, beta)
            elif self.aggregation.__name__ == 'median' or self.aggregation == 'geometric_median':
                aggregation_res = self.aggregation(grads_bf_comm)
            else:
                aggregation_res = self.aggregation(grads_bf_comm, self.byzantine_size)
            if iteration % self.display_interval == 0:
                print(aggregation_res)
            dist_models.activate_model(-1)  # 激活中心模型
            model = dist_models.model
            cumulated_param = 0
            with torch.no_grad():
                for param in model.parameters():
                    param_size = param.nelement()
                    beg, end = cumulated_param, cumulated_param + param.nelement()
                    cumulated_param += param_size
                    param.data.mul_(1 - self.weight_decay * lr)  # 1 - self.weight_decay * lr
                    # param.data.sub_(param.grad, alpha=lr)
                    param.data.sub_(aggregation_res[beg:end].view_as(param), alpha=lr)  # alpha=lr


            for node in self.nodes:
                dist_models.params_vec[node].copy_(dist_models.param_central)
            # 把test换到这里
            if iteration % self.display_interval == 0:
                if flag == 1:
                    loss_avg, loss_var = worst_loss_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    display = '[vanillaCSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
                    'avg loss={:.3e}, var loss={:.4f}, lr={:f}'
                    log(display.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, loss_var, lr
                    ))

                else:
                    loss_avg, acc_avg, acc_worst, var_acc = worst_loss_accuracy_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    loss_path.append(loss_avg)
                    acc_path.append(acc_avg)
                    worst_acc_path.append(acc_worst)
                    var_path.append(var_acc)

                    log(hint.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, acc_avg, acc_worst, var_acc, lr
                    ))
        loss_path = [x.tolist() for x in loss_path]
        acc_path = [x.tolist() for x in acc_path]
        print('done')
        return 0, loss_path, acc_path, worst_acc_path, var_path



class qFFL(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(name='qFFL', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation

    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)

        # initial record
        loss_path = []
        acc_path = []
        worst_acc_path = []
        var_path = []
        flag = 0  # 记录是否线性，线性为1
        if type(self.model).__name__ == 'LeastSqaure_LinearModel':
            flag = 1
        q = 5 

        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[qFFL]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, avg acc={:.4f}, worst acc={:.4f}, var={:.4e}, lr={:f}'
        # local models before communication
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack)
                      for node in self.nodes]# if node in self.honest_nodes else None for node in self.nodes

        for iteration in range(0, self.total_iterations + 1):
            hs = [0 for _ in self.nodes]
            test_iters = [self.get_test_iter(dataset=self.dist_test_set[node], rng_pack=self.rng_pack)
                          #if node in self.honest_nodes else None
                          for node in self.nodes] # if node in self.honest_nodes else None for node in self.nodes
            #lr = self.lr_ctrl.get_lr(iteration)
            lr = 10
            # lr=0.9
            loss_last_round = dist_models.loss_vec.clone()
            # lastest learning rate
            # test本来是在这里的
            # Collect gradients from honest nodes
            for node in self.nodes: #好人执行常规操作#for node in self.graph.honest_nodes
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                dist_models.loss_vec[node] = loss  # 更新当前轮次的loss
                model.zero_grad()
                loss.backward()

                cumulated_param = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.nelement()
                        beg, end = cumulated_param, cumulated_param + param.nelement()
                        cumulated_param += param_size
                        dist_models.grads_vec[node][beg:end] = param.grad.view(-1)
                    dist_models.grads_vec[node].view_as(dist_models.params_vec[node])
                grad_norm = torch.norm(dist_models.grads_vec[node])
                hs[node] = q * torch.pow(loss + 1e-10, q - 1) * (grad_norm ** 2) + \
                           (1.0 / lr) * torch.pow(loss + 1e-10, q)


            grads_bf_comm = dist_models.grads_vec.clone()
            combined_grads = dist_models.grads_vec.clone()
            for node in self.nodes:
                if node in self.honest_nodes:
                    combined_grads[node] = torch.pow(dist_models.loss_vec[node] + 1e-10, q) * grads_bf_comm[node]
            loss_honest = [loss_last_round[node] for node in self.graph.honest_nodes]


            if iteration % self.display_interval == 0:
                if flag != 1:
                    for node in self.nodes:
                        print('node: {}, loss: {}'.format(node, dist_models.loss_vec[node]))

            if self.attack != None:
                if self.attack.__name__ == 'disguise':
                    for node in self.byzantine_nodes:
                        grads_bf_comm[node] = grads_bf_comm[4].clone()
                        dist_models.loss_vec[node] = dist_models.loss_vec[4]
                for node in self.byzantine_nodes:
                    combined_grads[node] = torch.pow(dist_models.loss_vec[node] + 1e-10, q) * grads_bf_comm[node]
                if self.attack.__name__ == 'sign_flipping':
                    self.attack(combined_grads, self.honest_nodes, self.byzantine_nodes, 20, 0)
                elif self.attack.__name__ == 'gaussian':
                    self.attack(combined_grads, self.honest_nodes, self.byzantine_nodes, 20)
            beta = 0 if self.attack == None else 0.2
            # aggregation
            dt = 0.1
            if iteration > 2000:
                dt = 0.1
            if self.aggregation.__name__ == 'mean':
                aggregation_res = self.aggregation(combined_grads) * self.node_size / (torch.sum(torch.tensor(hs)))
            elif self.aggregation.__name__ == 'NBS':
                denominator, aggregation_res = NBS(combined_grads, hs, beta)
                aggregation_res = aggregation_res * self.node_size / denominator
            elif self.aggregation.__name__ == 'median' or self.aggregation == 'geometric_median':

                aggregation_res = self.aggregation(combined_grads) * self.node_size / (dt * torch.sum(torch.tensor(hs)))
            else:
                aggregation_res = self.aggregation(combined_grads, self.byzantine_size) * self.node_size / (torch.sum(torch.tensor(hs)))

            # 在这里进行attack，对于Delta和hs进行修改
            dist_models.activate_model(-1)  # Activate central model
            model = dist_models.model
            cumulated_param = 0
            with torch.no_grad():
                for param in model.parameters():
                    param_size = param.nelement()
                    beg, end = cumulated_param, cumulated_param + param.nelement()
                    cumulated_param += param_size
                    param.data.sub_(aggregation_res[beg:end].view_as(param), alpha=1)  # alpha=lr 

            # communication and attack
            for node in self.nodes:#for node in self.graph.honest_nodes:
                dist_models.params_vec[node].copy_(dist_models.param_central)
            # 把test换到这里
            if iteration % self.display_interval == 0:
                if flag == 1:
                    loss_avg, loss_var = worst_loss_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    display = '[qFFL]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
                              'avg loss={:.3e}, var loss={:.4f}, lr={:f}'
                    log(display.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, loss_var, lr
                    ))
                else:
                    loss_avg, acc_avg, acc_worst, var_acc = worst_loss_accuracy_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    loss_path.append(loss_avg)
                    acc_path.append(acc_avg)
                    worst_acc_path.append(acc_worst)
                    var_path.append(var_acc)
                    # var_path.append(var_loss)

                    log(hint.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, acc_avg, acc_worst, var_acc, lr
                    ))

        loss_path = [x.tolist() for x in loss_path]
        acc_path = [x.tolist() for x in acc_path]

        print('----end-----')
        return 0, loss_path, acc_path, worst_acc_path, var_path



class DRFL(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(name='DRFL', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation

    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)

        # initial record
        loss_path = []
        acc_path = []
        worst_acc_path = []
        var_path = []
        flag = 0  # 记录是否线性，线性为1
        if type(self.model).__name__ == 'LeastSqaure_LinearModel':
            flag = 1
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DRFL]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, avg acc={:.4f}, worst acc={:.4f}, var={:.4e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        loss_last_round = torch.zeros_like(dist_models.loss_vec)
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack)
                      for node in self.nodes]# if node in self.honest_nodes else None for node in self.nodes]

        for iteration in range(0, self.total_iterations + 1):
            test_iters = [self.get_test_iter(dataset=self.dist_test_set[node], rng_pack=self.rng_pack)
                          if node in self.honest_nodes else None
                          for node in self.nodes]
            # lastest learning rate
            lr = 0.01 
            #lr = self.lr_ctrl.get_lr(iteration)
            param_bf_comm.copy_(dist_models.params_vec)
            loss_last_round.copy_(dist_models.loss_vec)

            # gradient descent
            for node in self.nodes:#self.graph.honest_nodes
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                dist_models.loss_vec[node] = loss  # 更新当前轮次的loss
                model.zero_grad()
                loss.backward()
                cumulated_param = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.nelement()
                        beg, end = cumulated_param, cumulated_param + param.nelement()
                        cumulated_param += param_size
                        dist_models.grads_vec[node][beg:end] = param.grad.view(-1)
                    dist_models.grads_vec[node].view_as(dist_models.params_vec[node])
            if iteration % self.display_interval == 0:
                print(dist_models.grads_vec[0])

            grads_bf_comm = dist_models.grads_vec.clone()
            combined_grads = dist_models.grads_vec.clone()

            loss_honest = [loss_last_round[node] for node in self.graph.honest_nodes]
            loss_last_round_norm = loss_last_round / loss_last_round.sum()
            weight = [1 / self.node_size for _ in range(self.node_size)]
            for j in range(self.node_size):
                weight[j] = loss_last_round_norm[j].item()
            if iteration % self.display_interval == 0:
                print('weight: {}'.format(weight))
            if iteration % self.display_interval == 0:
                for node in self.nodes:
                    print('node: {}, loss: {}'.format(node, dist_models.loss_vec[node]))
            # attack
            for node in self.nodes:
                if node in self.honest_nodes:
                    combined_grads[node] = grads_bf_comm[node] * torch.tensor(weight[node]) * self.node_size
            if self.attack != None:
                if self.attack.__name__ == 'disguise':
                    for node in self.byzantine_nodes:
                        #tar = random.choice([0, 1, 2, 3, 5, 6, 7])
                        grads_bf_comm[node] = grads_bf_comm[4].clone()
                for node in self.byzantine_nodes:
                    combined_grads[node] = grads_bf_comm[node] * torch.tensor(weight[node]) * self.node_size
                if self.attack.__name__ == 'sign_flipping':
                    self.attack(combined_grads, self.honest_nodes, self.byzantine_nodes, 20, 0)
                elif self.attack.__name__ == 'gaussian':
                    self.attack(combined_grads, self.honest_nodes, self.byzantine_nodes, 2000)

            # aggregation
            beta = 0 if self.attack == None else 0.2
            if self.aggregation.__name__ == 'mean':
                aggregation_grad = self.aggregation(combined_grads)
            elif self.aggregation.__name__ == 'NBS_AFL':
                aggregation_grad = self.aggregation(combined_grads, beta)
            elif self.aggregation.__name__ == 'median' or self.aggregation == 'geometric_median':
                aggregation_grad = self.aggregation(combined_grads)
            else:
                aggregation_grad = self.aggregation(combined_grads, self.byzantine_size)

            dist_models.activate_model(-1)  # 激活中心模型
            model = dist_models.model
            cumulated_param = 0
            # gradient descend
            with torch.no_grad():
                for param in model.parameters():
                    param_size = param.nelement()
                    beg, end = cumulated_param, cumulated_param + param.nelement()
                    cumulated_param += param_size
                    #param.data.mul_(1 - self.weight_decay * lr)  # 1 - self.weight_decay * lr
                    # param.data.sub_(param.grad, alpha=lr)
                    param.data.sub_(aggregation_grad[beg:end].view_as(param), alpha=lr)  # alpha=lr
                    #param.data.mul_(1 - self.weight_decay * lr2)  # 应该是防止过拟合的

            for node in self.nodes:
                dist_models.params_vec[node].copy_(dist_models.param_central)
                #dist_models.params_lambda_vec[node].copy_(aggregation_lambda)
            # 把test换到这里
            if iteration % self.display_interval == 0:
                if flag == 1:
                    loss_avg, loss_var = worst_loss_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    display = '[DRFL]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
                              'avg loss={:.3e}, var loss={:.4f}, lr={:f}'
                    log(display.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, loss_var, lr
                    ))
                else:
                    loss_avg, acc_avg, acc_worst, var_acc = worst_loss_accuracy_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    loss_path.append(loss_avg)
                    acc_path.append(acc_avg)
                    worst_acc_path.append(acc_worst)
                    var_path.append(var_acc)
                    # var_path.append(var_loss)

                    log(hint.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, acc_avg, acc_worst, var_acc, lr
                    ))

        loss_path = [x.tolist() for x in loss_path]
        acc_path = [x.tolist() for x in acc_path]

        print('----end-----')
        return 0, loss_path, acc_path, worst_acc_path, var_path



class AFL(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(name='AFL', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation

    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)

        # initial record
        loss_path = []
        acc_path = []
        worst_acc_path = []
        var_path = []
        flag = 0  # 记录是否线性，线性为1
        if type(self.model).__name__ == 'LeastSqaure_LinearModel':
            flag = 1
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[AFL]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, avg acc={:.4f}, worst acc={:.4f}, var={:.4e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        loss_last_round = torch.zeros_like(dist_models.loss_vec)
        #lambda_bf_comm = torch.zeros_like(dist_models.params_lambda_vec)
        #last_aggregation_res = torch.zeros_like(param_bf_comm[0]) # add
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack)
                      for node in self.nodes]# if node in self.honest_nodes else None for node in self.nodes]

        for iteration in range(0, self.total_iterations + 1):
            test_iters = [self.get_test_iter(dataset=self.dist_test_set[node], rng_pack=self.rng_pack)
                          for node in self.nodes]# if node in self.honest_nodes else None for node in self.nodes]
            # lastest learning rate
            lr = 0.01 #常数步长好！
            #lr = self.lr_ctrl.get_lr(iteration)
            lambda_last_round = dist_models.params_lambda
            param_bf_comm.copy_(dist_models.params_vec)
            loss_last_round.copy_(dist_models.loss_vec)
            if self.attack != None:
                if self.attack.__name__ == 'disguise':
                    for node in self.byzantine_nodes:
                        loss_last_round[node] = loss_last_round[4]
            grad_lambda = loss_last_round / loss_last_round.sum()
            lr1 = 5 * lr
            grad_bf_proj = [a + b for a, b in zip(lambda_last_round,
                                                  [lr1 * i for i in
                                                   grad_lambda.tolist()])]  # 0.4
            dist_models.params_lambda = project(grad_bf_proj)
            weight = [0 for _ in range(self.node_size)]
            for j in range(self.node_size):
                #if j in self.graph.honest_nodes:
                weight[j] = dist_models.params_lambda[j]
            if iteration % self.display_interval == 0:
                print('weight: {}'.format(weight))

            # gradient descent
            for node in self.nodes:#self.graph.honest_nodes
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                dist_models.loss_vec[node] = loss  # 更新当前轮次的loss
                model.zero_grad()
                loss.backward()
                cumulated_param = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.nelement()
                        beg, end = cumulated_param, cumulated_param + param.nelement()
                        cumulated_param += param_size
                        dist_models.grads_vec[node][beg:end] = param.grad.view(-1)
                    dist_models.grads_vec[node].view_as(dist_models.params_vec[node])
                if iteration % self.display_interval == 0:
                    print('node: {}, loss: {}'.format(node, loss))
            grads_bf_comm = dist_models.grads_vec.clone()
            combined_grads = dist_models.grads_vec.clone()

            loss_honest = [loss_last_round[node] for node in self.graph.honest_nodes]
            # attack for loss
            for node in self.nodes:
                if node in self.honest_nodes:
                    combined_grads[node] = grads_bf_comm[node] * torch.tensor(weight[node]) * self.node_size
            if iteration % self.display_interval == 0:
                print(combined_grads)
            if self.attack != None:
                if self.attack.__name__ == 'disguise':
                    for node in self.byzantine_nodes:
                        grads_bf_comm[node] = grads_bf_comm[4].clone()
                for node in self.byzantine_nodes:
                    combined_grads[node] = grads_bf_comm[node] * torch.tensor(weight[node]) * self.node_size
                if self.attack.__name__ == 'sign_flipping':
                    self.attack(combined_grads, self.honest_nodes, self.byzantine_nodes, 20, 0)
                elif self.attack.__name__ == 'gaussian':
                    self.attack(combined_grads, self.honest_nodes, self.byzantine_nodes, 20)

            # aggregation
            beta = 0 if self.attack == None else 0.2
            if self.aggregation.__name__ == 'mean':
                aggregation_grad = self.aggregation(combined_grads)
            elif self.aggregation.__name__ == 'NBS_AFL':
                aggregation_grad = self.aggregation(combined_grads, beta)
            elif self.aggregation.__name__ == 'median' or self.aggregation == 'geometric_median':
                aggregation_grad = self.aggregation(combined_grads)
            else:
                aggregation_grad = self.aggregation(combined_grads, self.byzantine_size)

            if iteration % self.display_interval == 0:
                print(aggregation_grad)
            dist_models.activate_model(-1)  # 激活中心模型
            model = dist_models.model
            cumulated_param = 0
            # gradient descend
            with torch.no_grad():
                for param in model.parameters():
                    param_size = param.nelement()
                    beg, end = cumulated_param, cumulated_param + param.nelement()
                    cumulated_param += param_size
                    param.data.sub_(aggregation_grad[beg:end].view_as(param), alpha=lr)  # alpha=lr

            for node in self.nodes:
                dist_models.params_vec[node].copy_(dist_models.param_central)
            # 把test换到这里
            if iteration % self.display_interval == 0:
                if flag == 1:
                    loss_avg, loss_var = worst_loss_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    display = '[AFL]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
                              'avg loss={:.3e}, var loss={:.4f}, lr={:f}'
                    log(display.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, loss_var, lr
                    ))
                else:
                    loss_avg, acc_avg, acc_worst, var_acc = worst_loss_accuracy_dist(
                        dist_models, test_iters,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    loss_path.append(loss_avg)
                    acc_path.append(acc_avg)
                    worst_acc_path.append(acc_worst)
                    var_path.append(var_acc)

                    log(hint.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_avg, acc_avg, acc_worst, var_acc, lr
                    ))

        loss_path = [x.tolist() for x in loss_path]
        acc_path = [x.tolist() for x in acc_path]
        return 0, loss_path, acc_path, worst_acc_path, var_path
    
class Hnobs(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(name='Hnobs', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation

    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)

        # initial record
        loss_path = []
        acc_path = []
        worst_acc_path = []
        var_path = []
        q = 2 #qFFL文章中说q取5和15

        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[Hnobs]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, avg acc={:.4f}, worst acc={:.4f}, var={:.4e}, lr={:f}'
        # local models before communication
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack)#if node in self.honest_nodes else None for node in self.nodes
                      for node in self.nodes]

        for iteration in range(0, self.total_iterations + 1):
            hs = [0 for _ in self.nodes]
            test_iters = [self.get_test_iter(dataset=self.dist_test_set[node], rng_pack=self.rng_pack)#if node in self.honest_nodes else None for node in self.nodes
                          for node in self.nodes]
            #lr = self.lr_ctrl.get_lr(iteration)
            lr = 0.9
            # lastest learning rate
            loss_last_round = dist_models.loss_vec.clone()
            # Collect gradients from honest nodes
            for node in self.nodes: #for node in self.graph.honest_nodes
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                dist_models.loss_vec[node] = loss  # 更新当前轮次的loss
                model.zero_grad()
                loss.backward()

                cumulated_param = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.nelement()
                        beg, end = cumulated_param, cumulated_param + param.nelement()
                        cumulated_param += param_size
                        dist_models.grads_vec[node][beg:end] = param.grad.view(-1)
                    dist_models.grads_vec[node].view_as(dist_models.params_vec[node])
                grad_norm = torch.norm(dist_models.grads_vec[node])
                hs[node] = q * torch.pow(loss + 1e-10, q - 1) * (grad_norm ** 2) + \
                           (1.0 / lr) * torch.pow(loss + 1e-10, q)

            grads_bf_comm = dist_models.grads_vec.clone()
            combined_grads = dist_models.grads_vec.clone()

            for node in self.honest_nodes:
                combined_grads[node] = torch.pow(dist_models.loss_vec[node] + 1e-10, q) * grads_bf_comm[node]
            if self.attack != None:
                if self.attack.__name__ == 'sign_flipping':
                    self.attack(grads_bf_comm, self.honest_nodes, self.byzantine_nodes, 1, 0)
                elif self.attack.__name__ == 'gaussian':
                    self.attack(grads_bf_comm, self.honest_nodes, self.byzantine_nodes, 5)
                else:
                    for node in self.byzantine_nodes:
                        dist_models.loss_vec[node] = max([loss_last_round[i] for i in self.honest_nodes])
                        grads_bf_comm[node] = grads_bf_comm[0].clone()
                for node in self.byzantine_nodes:
                    combined_grads[node] = torch.pow(dist_models.loss_vec[node] + 1e-10, q) * grads_bf_comm[node]

            beta = 0 if self.attack == None else 0.2
            # aggregation
            if callable(self.aggregation):
                if self.aggregation.__name__  == 'NBS':
                    denominator, aggregation_res = NBS(combined_grads, hs, beta)
                    aggregation_res = aggregation_res * self.node_size / denominator
                elif self.aggregation.__name__  == 'median' or self.aggregation == 'geometric_median':
                    aggregation_res = self.aggregation(combined_grads) * self.node_size / (0.8 * sum(hs))
                else:
                    aggregation_res = self.aggregation(combined_grads, self.byzantine_size) * self.node_size / (0.8 * sum(hs))
            else:
                aggregation_res = self.aggregation.run(grads_bf_comm)

            # 在这里进行attack，对于Delta和hs进行修改
            dist_models.activate_model(-1)  # Assuming 0-th node is used for averaging
            model = dist_models.model
            cumulated_param = 0
            with torch.no_grad():
                for param in model.parameters():
                    param_size = param.nelement()
                    beg, end = cumulated_param, cumulated_param + param.nelement()
                    cumulated_param += param_size
                    param.data.sub_(aggregation_res[beg:end].view_as(param), alpha=1)  # alpha=lr 在qFFL论文中，这里是直接梯度下降的，没有步长

            for node in self.nodes:
                dist_models.params_vec[node].copy_(dist_models.param_central)
            # 把test换到这里
            if iteration % self.display_interval == 0:
                loss_avg, acc_avg, acc_worst, var_acc = worst_loss_accuracy_dist(
                    dist_models, test_iters,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                loss_path.append(loss_avg)
                acc_path.append(acc_avg)
                worst_acc_path.append(acc_worst)
                var_path.append(var_acc)

                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    loss_avg, acc_avg, acc_worst, var_acc, lr
                ))

        loss_path = [x.tolist() for x in loss_path]
        acc_path = [x.tolist() for x in acc_path]

        print('----end-----')
        return 0, loss_path, acc_path, worst_acc_path, var_path
