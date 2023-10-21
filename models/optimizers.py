import torch
import torch.optim as optim
# from ranger import RangerQH, Ranger
# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/rangerqh.py
# from ranger21.ranger21 import Ranger21


def get_params_groups(net, lr, lr_multi=10):
    if hasattr(net, 'trainable_parameters'):
        param_groups = net.trainable_parameters()
    else:
        parameters = filter(lambda p: p.requires_grad, net.parameters())   # 适配frozen layers的情况
        param_groups = list(parameters)
        # print(f'layers that requires grad')
        # for k, v in net.named_parameters():
        #     if v.requires_grad:
        #         print(k)
    if isinstance(param_groups, list):
        params_list = [{'params': param_groups, 'lr': lr_multi * lr}]
    elif isinstance(param_groups, tuple):
        params_list = [{'params': param_groups[0]},
                       {'params': param_groups[1], 'lr': lr_multi * lr}]
    else:
        raise NotImplementedError
    return params_list


def get_optimizer(model_params, lr, optim_mode='sgd', lr_policy='linear', init_step=0, max_step=None):
    # if lr_policy != 'poly':
    if optim_mode == 'sgd':
        optimizer_G = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optim_mode == 'ranger':
        optimizer_G = RangerQH(model_params, lr=lr, betas=(0.9, 0.999))
    elif optim_mode == 'adam':
        optimizer_G = optim.Adam(model_params, lr=lr, betas=(0.9, 0.999))
    elif optim_mode == 'adamw':
        optimizer_G = optim.AdamW(model_params, lr=lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError()
    # else:
    #     if optim_mode == 'sgd':
    #         optimizer_G = PolyOptimizer(model_params, lr=lr,
    #                                     init_step=init_step, max_step=max_step,
    #                                     momentum=0.9, weight_decay=5e-4)
    #     elif optim_mode == 'adam':
    #         optimizer_G = PolyAdamOptimizer(model_params, lr=lr, betas=(0.9, 0.999),
    #                                         max_step=max_step, init_step=init_step,)
    #     elif optim_mode == 'adamw':
    #         optimizer_G = PolyAdamWOptimizer(model_params, lr=lr, betas=(0.9, 0.999),
    #                                         max_step=max_step, init_step=init_step,)
    #     elif optim_mode == 'ranger':
    #         optimizer_G = PolyRangerOptimizer(model_params, lr=lr, betas=(0.9, 0.999),
    #                                           init_step=init_step, max_step=max_step)
    #     # elif optim_mode == 'ranger21':
    #     #     optimizer_G = PolyRanger21Optimizer(model_params, lr=lr, betas=(0.9, 0.999),
    #     #                                         init_step=init_step, max_step=max_step)
    #     else:
    #         raise NotImplementedError

    return optimizer_G


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = init_step
        # print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def update_state_(self, init_step, max_step):
        self.global_step = init_step
        self.max_step = max_step

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


# class PolyRangerOptimizer(RangerQH):
#     def __init__(self, params, lr, betas, max_step, init_step=0, momentum=0.9):
#         super().__init__(params, lr, betas)
#
#         self.global_step = init_step
#         self.max_step = max_step
#         self.momentum = momentum
#
#         self.__initial_lr = [group['lr'] for group in self.param_groups]
#
#     def update_state_(self, init_step, max_step):
#         self.global_step = init_step
#         self.max_step = max_step
#
#     def step(self, closure=None):
#
#         if self.global_step < self.max_step:
#             lr_mult = (1 - self.global_step / self.max_step) ** self.momentum
#
#             for i in range(len(self.param_groups)):
#                 self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
#
#         super().step(closure)
#         self.global_step += 1


class PolyAdamOptimizer(torch.optim.Adam):
    def __init__(self, params, lr, betas, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, betas)

        self.global_step = init_step
        print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def update_state_(self, init_step, max_step):
        self.global_step = init_step
        self.max_step = max_step

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyAdamWOptimizer(torch.optim.AdamW):
    def __init__(self, params, lr, betas, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, betas)

        self.global_step = init_step
        print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def update_state_(self, init_step, max_step):
        self.global_step = init_step
        self.max_step = max_step

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

# class PolyRanger21Optimizer(Ranger21):
#     def __init__(self, params, lr, betas, max_step, init_step=0, momentum=0.9):
#         super().__init__(params, lr, betas=betas)
#
#         self.global_step = init_step
#         self.max_step = max_step
#         self.momentum = momentum
#
#         self.__initial_lr = [group['lr'] for group in self.param_groups]
#
#     def update_state_(self, init_step, max_step):
#         self.global_step = init_step
#         self.max_step = max_step
#
#     def step(self, closure=None):
#
#         if self.global_step < self.max_step:
#             lr_mult = (1 - self.global_step / self.max_step) ** self.momentum
#
#             for i in range(len(self.param_groups)):
#                 self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
#
#         super().step(closure)
#         self.global_step += 1
