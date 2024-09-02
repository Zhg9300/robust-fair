import numpy as np
import torch

from ByrdLab import TARGET_TYPE

@torch.no_grad()
def consensus_error(local_models, honest_nodes):
    return torch.var(local_models[honest_nodes], 
                     dim=0, unbiased=False).norm().item()

@torch.no_grad()
def avg_loss(model, get_test_iter, loss_fn, weight_decay):
    '''
    The function calculating the loss.
    '''
    loss = 0
    total_sample = 0
    
    # evaluation
    test_iter = get_test_iter()
    for features, targets in test_iter:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        total_sample += len(targets)
    
    loss_avg = loss / total_sample
    for param in model.parameters():
        loss_avg += weight_decay * param.norm()**2 / 2

    return loss_avg
    
@torch.no_grad()
def avg_loss_accuracy(model, get_test_iter, loss_fn, weight_decay):
    '''
    The function calculating the loss and accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    test_iter = get_test_iter()
    for features, targets in test_iter:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        _, prediction_cls = torch.max(predictions.detach(), dim=1)
        accuracy += (prediction_cls == targets).sum().item()
        total_sample += len(targets)
    
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample
    for param in model.parameters():
        loss_avg += weight_decay * param.norm()**2 / 2

    return loss_avg, accuracy_avg

@torch.no_grad()
def binary_classification_accuracy(predictions, targets):
    prediction_cls = (predictions > 0.5).type(TARGET_TYPE)
    accuracy = (prediction_cls == targets).sum()
    return accuracy

@torch.no_grad()
def multi_classification_accuracy(predictions, targets):
    _, prediction_cls = torch.max(predictions, dim=1)
    accuracy = (prediction_cls == targets).sum()
    return accuracy

@torch.no_grad()
def avg_loss_accuracy_dist(dist_models, get_test_iter,
                           loss_fn, test_fn, weight_decay,
                           node_list=None):
    '''
    The function calculating the loss and accuracy in distributed setting.
    Return the avarage of the local models' accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    # dist_models.activate_avg_model()
    if node_list is None:
        node_list = range(dist_models.node_size)
    for node in node_list:
        dist_models.activate_model(node)
        model = dist_models.model
        test_iter = get_test_iter()
        for features, targets in test_iter:
            predictions = model(features)
            loss += loss_fn(predictions, targets).item()
            accuracy += test_fn(predictions, targets).item()
            total_sample += len(targets)
        
        penalization = 0
        for param in model.parameters():
            penalization += weight_decay * param.norm()**2 / 2
        loss += penalization
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg

@torch.no_grad()
def worst_loss_dist(dist_models, test_iters, loss_fn,
                             test_fn, weight_decay, node_list=None):
    # evaluation
    if node_list is None:
        node_list = range(dist_models.node_size)
    loss_list = []

    for node in node_list:
        features, targets = next(test_iters[node])
        # print('node:',node)
        # print('targets',targets)
        #test_iter = get_test_iter() # TODO 这里的testset对所有的节点都没区别，这是需要修改的
        dist_models.activate_model(node)
        model = dist_models.model
        predictions = model(features)
        #print('pre  ', predictions)
        loss_this_node = loss_fn(predictions, targets).item()
        penalization = 0
        for param in model.parameters():
            penalization += weight_decay * param.norm() ** 2 / 2
        loss_this_node += penalization.item()
        sample_size_this_node = len(targets)
        #print(node, '    ', sample_size_this_node)
        loss_this_node = loss_this_node / sample_size_this_node
        loss_list.append(loss_this_node)
    print(loss_list)
    loss_avg = np.mean(loss_list)
    loss_var = np.var(loss_list)
    return loss_avg, loss_var


@torch.no_grad()
def worst_loss_accuracy_dist(dist_models, test_iters, loss_fn,
                             test_fn, weight_decay, node_list=None):
    loss = 0
    total_sample = 0
    #device = torch.device("cuda")
    # evaluation
    # TODO: debug
    if node_list is None:
        node_list = range(dist_models.node_size)
    accuracy_list = []
    loss_list = []

    for node in node_list:
        features, targets = next(test_iters[node])
        # print('node:',node)
        # print('targets',targets)
        #test_iter = get_test_iter() # TODO 这里的testset对所有的节点都没区别，这是需要修改的
        dist_models.activate_model(node)
        model = dist_models.model
        predictions = model(features)
        loss += loss_fn(predictions, targets).item()
        accuracy = test_fn(predictions, targets).item()  # 对一个batch中的样本预测结果进行统计
        total_sample += len(targets)
        sample_size_this_node = len(targets)
        acc_avg = accuracy / sample_size_this_node
        #print('node {:f}, sample size {:f}'.format(node, sample_size_this_node))
        accuracy_list.append(acc_avg)
        penalization = 0
        for param in model.parameters():
            penalization += weight_decay * param.norm() ** 2 / 2
        loss += penalization
        #loss_avg = loss / sample_size_this_node
        #loss_list.append(loss_avg)
    print(accuracy_list)
    loss_avg = loss / total_sample
    accuracy_avg = np.mean(accuracy_list)
    accuracy_worst = min(accuracy_list)
    var_acc = np.var(accuracy_list)
    #var_loss = np.var(loss_list)
    #loss的部分先不改动
    return loss_avg, accuracy_avg, accuracy_worst, var_acc

@torch.no_grad()
def rsa_worst_loss_accuracy_dist(dist_models, test_iters, loss_fn,
                             test_fn, weight_decay, node_list=None):
    loss = 0
    total_sample = 0
    #device = torch.device("cuda")
    # evaluation
    # TODO: debug
    if node_list is None:
        node_list = range(dist_models.node_size)
    accuracy_list = []

    for node in node_list:
        features, targets = next(test_iters[node])
        #test_iter = get_test_iter() # TODO 这里的testset对所有的节点都没区别，这是需要修改的
        accuracy = 0
        sample_size_this_node = 0
        dist_models.activate_model(0)# 0号节点暂存了中心的模型
        model = dist_models.model
        predictions = model(features)
        loss += loss_fn(predictions, targets).item()
        accuracy += test_fn(predictions, targets).item()  # 对一个batch中的样本预测结果进行统计
        total_sample += len(targets)
        sample_size_this_node += len(targets)
        acc_avg = accuracy / sample_size_this_node
        #print('node {:f}, sample size {:f}'.format(node, sample_size_this_node))
        accuracy_list.append(acc_avg)
        penalization = 0
        for param in model.parameters():
            penalization += weight_decay * param.norm() ** 2 / 2
        loss += penalization
    print(accuracy_list)
    loss_avg = loss / total_sample
    accuracy_avg = np.mean(accuracy_list)
    accuracy_worst = min(accuracy_list)
    var_acc = np.var(accuracy_list)
    #loss的部分先不改动
    return loss_avg, accuracy_avg, accuracy_worst, var_acc
