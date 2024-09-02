
from ByrdLab import FEATURE_TYPE
import torch
from torch.nn.parameter import Parameter
from ByrdLab.library.tool import get_model_param

AVG_MODEL = -1

class DistributedModule():
   
    def __init__(self, model, node_size):
        # 将原先的avgmodel换成centralmodel！
        self.model = model
        self.active_node = -1
        self.CENTRAL_MODEL_INDEX = node_size
        self.node_size = node_size

        # model parameters in the form of tensor with size [node_size x mdoel_size]
        # The last line is the average model
        model_size = get_model_param(model, use_str=False)
        self.model_size = model_size
        print('model_size', model_size)
        self.params_vec = torch.zeros([node_size, model_size],
                                      dtype=FEATURE_TYPE) #torch.float64
        #self.param_avg = torch.zeros(model_size, dtype=FEATURE_TYPE)
        #self.grads_vec = None
        self.grads_vec = torch.zeros([node_size, model_size],
                                      dtype=FEATURE_TYPE)
        
        # model parameters in the form of torch.nn.parameter.Parameter
        self.params_torch = []
        # --------------added by zhg
        #self.grads_torch = [] # grads_torch的相关都是添加的
        # 有中心的情况下，只需要中心来存储loss即可
        # self.params_lambda_vec = [[] for _ in range(node_size)]  # 存储所有的lambda，初始值为空列表
        # self.loss_vec = torch.tensor(1e-5).repeat(node_size)
        self.param_central = torch.zeros(model_size, dtype=FEATURE_TYPE)
        self.params_lambda = [1 / node_size for _ in range(node_size)]
        self.loss_vec = torch.tensor(1e-5).repeat(node_size)
        # --------------------------
        for node in range(node_size+1):
            cumulated_param = 0
            param_list = []
            for param in model.parameters():
                param_size = param.nelement()
                beg, end = cumulated_param, cumulated_param + param.nelement()
                cumulated_param += param_size
                if node != self.CENTRAL_MODEL_INDEX:
                    vec = self.params_vec[node][beg:end]
                    # # --------------added by zhg
                    # grad = self.grads_vec[node][beg:end]
                    # new_grads = Parameter(grad.view_as(param.grad), requires_grad=True)
                    # grads_list.append(new_grads)
                else:
                    vec = self.param_central[beg:end]
                    
                new_param = Parameter(vec.view_as(param), requires_grad=False)
                param_list.append(new_param)


            self.params_torch.append(param_list)
            #print(self.params_torch[0].shape)
        
    def activate_model(self, node):
        if self.active_node != node:
            old_named_parmas = self.model.named_parameters()
            new_params = self.params_torch[node]
            for (name, old_param), new_param in zip(old_named_parmas,
                                                    new_params):
                old_param.requires_grad = False
                # if node != self.CENTRAL_MODEL_INDEX:
                #     new_param.requires_grad = True
                new_param.requires_grad = True
                self.set_param(name, new_param)
            self.active_node = node
        
    def activate_central_model(self, node_list=None):
        #self.param_central.copy_(self.get_central_param(node_list))
        self.activate_model(self.CENTRAL_MODEL_INDEX)

        
    # def get_central_param(self, node_list=None):
    #     return self.param_central
        # if node_list is None:
        #     return self.params_vec.mean(axis=0)
        # else:
        #     avg_model = torch.zeros(self.model_size)
        #     for node in node_list:
        #         avg_model.add_(self.params_vec[node])
        #     return avg_model / len(node_list)
                
    def set_param(self, param_name, new_param):
        atoms = param_name.split(".")
        
        model = self.model
        for item in atoms[:-1]:
            model = getattr(model, item)
        setattr(model, atoms[-1], new_param)
        
    def norm(self, node, *args, **kw):
        return self.params_vec[node].norm(*args, **kw)
        
    # def norm_avg(self, node_list=None, *args, **kw):
    #     return self.get_avg_param(node_list=node_list).norm()
    
    # def init_grads_vec(self):
    #     self.avg_grad = torch.zeros(self.model_size, dtype=FEATURE_TYPE)
    #     self.grads_vec = torch.zeros_like(self.params_vec)
    #
    #     for node in range(self.node_size+1):
    #         cumulated_param = 0
    #         for (param_idx, param) in enumerate(self.model.parameters()):
    #             param_size = param.nelement()
    #             beg, end = cumulated_param, cumulated_param + param.nelement()
    #             cumulated_param += param_size
    #
    #             if node != self.CENTRAL_MODEL_INDEX:
    #                 grad = self.grads_vec[node][beg:end]
    #             else:
    #                 grad = self.param_central[beg:end]
    #
    #             grad_param = Parameter(grad.view_as(param), requires_grad=False)
    #             self.params_torch[node][param_idx].grad = grad_param
