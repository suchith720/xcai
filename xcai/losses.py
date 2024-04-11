# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_losses.ipynb.

# %% auto 0
__all__ = ['BaseLoss', 'MultiCrossEntropy', 'MultiTriplet', 'SoupCon']

# %% ../nbs/04_losses.ipynb 3
import functools, torch, torch.nn as nn, torch.nn.functional as F
from typing import MutableSequence, Union
from fastcore.utils import *
from fastcore.meta import *

from .torch_core import *

# %% ../nbs/04_losses.ipynb 13
class BaseLoss(nn.Module):

    def __init__(self, 
                 reduce:Optional[str]=None, 
                 **kwargs):
        super().__init__()
        self.reduce = reduce

    @property
    def reduction(self) -> str: return self.reduce
    
    @reduction.setter
    def reduction(self, v:str):
        "Sets the reduction style (typically 'mean', 'sum', or 'none')" 
        self.reduce = v
        

# %% ../nbs/04_losses.ipynb 15
class MultiCrossEntropy(BaseLoss):

    def __init__(self,
                 tn_targ:Optional[int]=None, 
                 ig_tok:Optional[int]=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.tn_targ, self.ig_tok = tn_targ, ig_tok
        self.o = torch.ones(tn_targ, dtype=torch.int64) if tn_targ is not None else None
        self._parameters = {'o': self.o}
        

# %% ../nbs/04_losses.ipynb 18
@patch
def __call__(cls:MultiCrossEntropy,
             inp:torch.FloatTensor,
             targ:torch.LongTensor,
             n_inp2targ:torch.LongTensor):
    tn_targ, targ_len = targ.shape
    bsz, inp_len, mn_targ = inp.shape[0], inp.shape[1], n_inp2targ.max()
    seq_len = min(targ_len, inp_len)
    inp, targ = -F.log_softmax(inp, dim=2)[:, :seq_len].transpose(1,2), targ[:, :seq_len]
    
    inp2targ_ptr = n_inp2targ.cumsum(dim=0)-1
    xn_inp2targ = mn_targ-n_inp2targ+1
    r_targ = (
        torch.ones(tn_targ, dtype=torch.int64, device=inp.device).scatter(0, inp2targ_ptr, xn_inp2targ)
        if cls.tn_targ is None or tn_targ > cls.tn_targ else
        cls.o[:tn_targ].scatter(0, inp2targ_ptr, xn_inp2targ)
    )
    xtarg = targ.repeat_interleave(r_targ, dim=0)

    s = inp.gather(1, xtarg.view(bsz, -1, seq_len)).view(-1, seq_len)
    s /= r_targ.repeat_interleave(r_targ, dim=0).view(-1, 1)
    idx = torch.where(xtarg != cls.ig_tok)
    loss = s[idx[0], idx[1]]
    
    if cls.reduction == 'mean': return (loss/len(torch.where(targ != cls.ig_tok)[0])).sum()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')


# %% ../nbs/04_losses.ipynb 25
class MultiTriplet(BaseLoss):

    def __init__(self,
                 bsz:Optional[int]=None, 
                 tn_targ:Optional[int]=None,
                 margin:Optional[float]=0.8,
                 ig_tok:Optional[int]=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.bsz, self.tn_targ, self.margin, self.ig_tok = bsz, tn_targ, margin, ig_tok
        self.t = torch.ones((bsz, bsz), dtype=torch.int64).triu() if bsz is not None else None
        self.u = torch.arange(bsz, dtype=torch.int64) if bsz is not None else None
        self.v = torch.ones(tn_targ, dtype=torch.int64) if tn_targ is not None else None
        self._parameters = {'t':self.t, 'u':self.u, 'v':self.v}
        

# %% ../nbs/04_losses.ipynb 28
@patch
def __call__(cls:MultiTriplet, 
             inp:torch.FloatTensor, 
             targ:torch.LongTensor, 
             n_inp2targ:torch.LongTensor,
             margin:Optional[float]=None):
    cls.margin = cls.margin if margin is None else margin
    bsz, tn_targ, mn_targ = inp.shape[0], targ.shape[0], n_inp2targ.max()
    t, u = cls.t[:bsz,:bsz], cls.u[:bsz]
    v = (
        torch.ones(tn_targ, dtype=torch.int64, device=targ.device)
        if tn_targ > cls.tn_targ else cls.v[:tn_targ]
    )
    targ2inp_ptr = u.repeat_interleave(n_inp2targ)
    s = targ@inp.T
    ps = s.gather(1, targ2inp_ptr.view(-1,1))
    
    inp2targ_ptr = CUDALongTensor.matmul(n_inp2targ[None], t).squeeze(0)-1
    xn_inp2targ = mn_targ-n_inp2targ+1
    
    r_targ = v.scatter(0, inp2targ_ptr, xn_inp2targ)
    
    targ2inp_ptrx = targ2inp_ptr.repeat_interleave(r_targ)
    mask, maskx = F.one_hot(targ2inp_ptr), F.one_hot(targ2inp_ptrx)
    fmask = CUDALongTensor.matmul(maskx,mask.T)
    psx = ps.repeat_interleave(r_targ).view(bsz, -1, 1)
    s = s.T.view(bsz, 1, -1)
    fs = (s - psx + cls.margin).view(-1, tn_targ)
    fs /= r_targ.repeat_interleave(r_targ).view(-1, 1)
    
    idx = torch.where(fmask == 0)
    loss = fs[idx[0], idx[1]]
    loss, n = torch.where(loss > 0, loss, 0), (n_inp2targ.sum())**2 - (n_inp2targ**2).sum()
    if cls.reduction == 'mean': return (loss/n).sum()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        

# %% ../nbs/04_losses.ipynb 33
class SoupCon(BaseLoss):

    @delegates(BaseLoss.__init__)
    def __init__(self,
                 bsz:Optional[int]=None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.t = torch.arange(bsz, dtype=torch.int64) if bsz is not None else None
        self._parameters = {'t':self.t}
        

# %% ../nbs/04_losses.ipynb 36
@patch
def __call__(cls:SoupCon,
             inp:torch.FloatTensor,
             targ:torch.LongTensor,
             n_inp2targ:torch.LongTensor):
    bsz = inp.shape[0]
    t = cls.t[:bsz]
    targ2inp_ptr = t.repeat_interleave(n_inp2targ)
    s = -F.log_softmax(targ@inp.T, dim=0)
    ps = s.gather(1, targ2inp_ptr.unsqueeze(1)).squeeze(1)
    if cls.reduce == 'mean':
        ps /= n_inp2targ.repeat_interleave(n_inp2targ)
        ps /= bsz
        return ps.sum()
    elif cls.reduce == 'sum': return ps.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        
