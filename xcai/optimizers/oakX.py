# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/28_optimizers.oakX.ipynb.

# %% auto 0
__all__ = ['MultipleOptimizer', 'MultipleScheduler']

# %% ../../nbs/28_optimizers.oakX.ipynb 2
import torch
from itertools import chain
from collections import defaultdict

# %% ../../nbs/28_optimizers.oakX.ipynb 3
class MultipleOptimizer(torch.optim.Optimizer):
    # Wrapper around multiple optimizers that should be executed at the same time
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def state(self):
        state = defaultdict(dict)
        for optimizer in self.optimizers:
            state = {**state, **optimizer.state}
        return state

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups = param_groups + optimizer.param_groups
        return param_groups

    def __getstate__(self):
        return [optimizer.__getstate__() for optimizer in self.optimizers]

    def __setstate__(self, state):
        for opt_state, optimizer in zip(self.optimizers, state):
            optimizer.__setstate__(opt_state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for optimizer in self.optimizers:
            format_string += '\n'
            format_string += optimizer.__repr__()
        format_string += ')'
        return format_string

    def _hook_for_profile(self):
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict):
        for state, optimizer in zip(state_dict, self.optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        raise NotImplementedError()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss


# %% ../../nbs/28_optimizers.oakX.ipynb 4
class MultipleScheduler(object):

    def __init__(self, sched):
        self.schedulers = sched

    def step(self, *args, **kwargs):
        for sched in self.schedulers: sched.step(*args, **kwargs)

    def get_last_lr(self):
        return list(chain(*[s.get_last_lr() for s in self.schedulers]))

    def state_dict(self):
        return [sched.state_dict() for sched in self.schedulers]

    def load_state_dict(self, state_dict):
        for sched,state in zip(self.schedulers, state_dict):
            sched.load_state_dict(state)
