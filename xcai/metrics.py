# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_metrics.ipynb.

# %% auto 0
__all__ = ['XCMetric', 'precision', 'Precision', 'recall', 'Recall', 'prec_recl', 'PrecRecl', 'mrr', 'Mrr', 'prec_recl_mrr',
           'PrecReclMrr', 'sort_xc_metrics']

# %% ../nbs/10_metrics.ipynb 3
import torch, numpy as np, scipy.sparse as sp
from collections import OrderedDict

from fastcore.utils import *
from fastcore.meta import *

import xclib.evaluation.xc_metrics as xm
from xclib.utils.sparse import rank, binarize

# %% ../nbs/10_metrics.ipynb 5
class XCMetric:

    def __init__(self, func, n_lbl:int, filterer:Optional[Union[np.array,sp.csr_matrix]]=None, **kwargs):
        self.func, self.n_lbl, self.filterer, self.kwargs = func, n_lbl, filterer, kwargs

    def reset(self):
        self.output = []

    def accumulate(self, **kwargs):
        self.output.append(kwargs)

    def __call__(self, **kwargs):
        self.reset()
        self.accumulate(**kwargs)
        return self.value

    def apply_filter(self, data):
        if self.filterer is not None:
            data[self.filterer[:,0], self.filterer[:,1]] = 0
            data.eliminate_zeros()
        return data

    def get_pred(self, output):
        data = (output['pred_score'], output['pred_idx'], output['pred_ptr'])
        pred = sp.csr_matrix(data, shape=(len(data[2])-1, self.n_lbl))
        pred.sum_duplicates()
        return self.apply_filter(pred)

    def get_targ(self, output):
        data = (torch.full((len(output['targ_idx']),), 1), output['targ_idx'], output['targ_ptr'])
        targ = sp.csr_matrix(data, shape=(len(data[2])-1, self.n_lbl))
        targ.sum_duplicates()
        return self.apply_filter(targ)
    
    @property
    def value(self):
        if len(self.output) == 0: return
        output = {k:torch.cat([o[k] for o in self.output]) for k in self.output[0]}
        output['targ_ptr'] = torch.cat([torch.tensor([0]), output['targ_ptr'].cumsum(dim=0)])
        output['pred_ptr'] = torch.cat([torch.tensor([0]), output['pred_ptr'].cumsum(dim=0)])
        
        pred, targ = self.get_pred(output), self.get_targ(output)
        return self.func(pred, targ, **self.kwargs)


# %% ../nbs/10_metrics.ipynb 6
def precision(inp:sp.csr_matrix, 
              targ:sp.csr_matrix, 
              prop:sp.csr_matrix=None, 
              k:Optional[int]=5, 
              pa:Optional[float]=0.55, 
              pb:Optional[float]=1.5, 
              repk:Optional[List]=None):
    
    name = ['P', 'N'] if prop is None else ['P', 'N', 'PSP', 'PSN']
    repk = [k] if repk is None else set(repk+[k])
    prop = None if prop is None else xm.compute_inv_propesity(prop, A=pa, B=pb)
    
    metric = xm.Metrics(true_labels=targ, inv_psp=prop)
    prec = metric.eval(inp, k)
    return {f'{n}@{r}': prec[i][r-1] for i,n in enumerate(name) for r in repk if r <= k}
    

# %% ../nbs/10_metrics.ipynb 7
@delegates(precision)
def Precision(n_lbl, filterer=None, **kwargs):
    return XCMetric(precision, n_lbl, filterer, **kwargs)
    

# %% ../nbs/10_metrics.ipynb 8
def recall(inp:sp.csr_matrix, 
           targ:sp.csr_matrix, 
           k:Optional[int]=5, 
           repk:Optional[List]=None):
    
    repk = [k] if repk is None else set(repk+[k])
    recl = xm.recall(inp, targ, k=k)
    return {f'R@{o}':recl[o-1] for o in repk if o <= k}
    

# %% ../nbs/10_metrics.ipynb 9
@delegates(precision)
def Recall(n_lbl, filterer=None, **kwargs):
    return XCMetric(recall, n_lbl, filterer, **kwargs)
    

# %% ../nbs/10_metrics.ipynb 10
def prec_recl(inp:sp.csr_matrix, 
              targ:sp.csr_matrix,
              prop:sp.csr_matrix=None,
              pa:Optional[float]=0.55,
              pb:Optional[float]=1.5,
              pk:Optional[int]=5,
              rep_pk:Optional[List]=None,
              rk:Optional[int]=5,
              rep_rk:Optional[List]=None):
    metric = precision(inp, targ, prop, k=pk, pa=pa, pb=pb, repk=rep_pk)
    metric.update(recall(inp, targ, k=rk, repk=rep_rk))
    return metric
    

# %% ../nbs/10_metrics.ipynb 11
@delegates(prec_recl)
def PrecRecl(n_lbl, filterer=None, **kwargs):
    return XCMetric(prec_recl, n_lbl, filterer, **kwargs)
    

# %% ../nbs/10_metrics.ipynb 13
def mrr(inp:sp.csr_matrix,
        targ:sp.csr_matrix,
        k:Optional[List]=[10]):
    ranks, targ = rank(inp), binarize(targ)
    metric = dict()
    for i in sorted(k, reverse=True):
        rs = ranks.copy()
        rs.data[rs.data > i] = 0.0
        rs.eliminate_zeros()
        rs = rs.multiply(targ)
        rs.data = 1.0/rs.data
        metric[f'MRR@{i}'] = rs.max(axis=1).mean()
    return metric

@delegates(mrr)
def Mrr(n_lbl, filterer=None, **kwargs):
    return XCMetric(mrr, n_lbl, filterer, **kwargs)
    

# %% ../nbs/10_metrics.ipynb 14
def prec_recl_mrr(inp:sp.csr_matrix,
        targ:sp.csr_matrix,
        prop:sp.csr_matrix=None,
        pa:Optional[float]=0.55,
        pb:Optional[float]=1.5,
        pk:Optional[int]=5,
        rep_pk:Optional[List]=None,
        rk:Optional[int]=5,
        rep_rk:Optional[List]=None,
        mk:Optional[List]=[10]):
    metric = prec_recl(inp, targ, prop=prop, pa=pa, pb=pb, pk=pk, rep_pk=rep_pk,
            rk=rk, rep_rk=rep_rk)
    metric.update(mrr(inp, targ, k=mk))
    return metric

@delegates(prec_recl_mrr)
def PrecReclMrr(n_lbl, filterer=None, **kwargs):
    return XCMetric(prec_recl_mrr, n_lbl, filterer, **kwargs)


# %% ../nbs/10_metrics.ipynb 15
def sort_xc_metrics(metric):
    order = {'P':1, 'N':2, 'PSP':3, 'PSN':4, 'R':5, 'PSR':6}
    def get_key(a,b): return (order.get(a,7), int(b)) 
    def sort_fn(k): return get_key(*k.split('@'))
    
    ord_metric = OrderedDict()
    for k in sorted(metric, key=sort_fn): metric[k] = ord_metric[k]
    return ord_metric
    
