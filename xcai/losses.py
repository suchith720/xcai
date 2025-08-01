# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_losses.ipynb.

# %% auto 0
__all__ = ['get_sparse_matrix', 'BaseLoss', 'MultiCrossEntropy', 'Calibration', 'BaseMultiTriplet', 'MultiTriplet',
           'MultiTripletFromInBatchScores', 'MultiTripletFromScores', 'MultiTripletWithNegatives',
           'MarginMSEWithNegatives', 'Cosine', 'Entropy', 'Triplet', 'SoupCon']

# %% ../nbs/04_losses.ipynb 3
import functools, torch, torch.nn as nn, torch.nn.functional as F, pickle
from typing import MutableSequence, Union, Tuple

from fastcore.utils import *
from fastcore.meta import *

from .torch_core import *
from .core import *

# %% ../nbs/04_losses.ipynb 15
def get_sparse_matrix(data_idx:torch.Tensor, n_data:torch.Tensor, scores:Optional[torch.Tensor]=None, 
                      size:Optional[Tuple]=None):
    data_ptr = torch.cat([torch.zeros(1, device=n_data.device, dtype=n_data.dtype), n_data.cumsum(0)])
    if scores is None: scores = torch.ones_like(data_idx)
    if data_idx.shape != scores.shape: raise ValueError(f'`data_idx` and `scores` should have same shape.')
    return (
        torch.sparse_csr_tensor(data_ptr, data_idx, scores, device=data_ptr.device)
        if size is None else
        torch.sparse_csr_tensor(data_ptr, data_idx, scores, device=data_ptr.device, size=size)
    )
    

# %% ../nbs/04_losses.ipynb 17
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
        

# %% ../nbs/04_losses.ipynb 19
class MultiCrossEntropy(BaseLoss):

    def __init__(self,
                 tn_targ:Optional[int]=None, 
                 ig_tok:Optional[int]=0,
                 vocab_weights:Optional[torch.Tensor]=None,
                 **kwargs):
        super().__init__(**kwargs)
        store_attr('tn_targ,ig_tok,vocab_weights')
        self.o = torch.ones(tn_targ, dtype=torch.int64) if tn_targ is not None else None
        

# %% ../nbs/04_losses.ipynb 21
@patch
def forward(cls:MultiCrossEntropy,
            inp:torch.FloatTensor,
            targ:torch.LongTensor,
            n_inp2targ:Optional[torch.LongTensor]=None,
            tn_targ:Optional[int]=None, 
            ig_tok:Optional[int]=None,
            vocab_weights:Optional[torch.Tensor]=None,
            **kwargs):
    store_attr('tn_targ,ig_tok,vocab_weights', is_none=False)
    
    cls.o = cls.o.to(inp.device) if cls.o is not None else None
    cls.vocab_weights = cls.vocab_weights.to(inp.device) if cls.vocab_weights is not None else None
    
    tn_targ, targ_len = targ.shape
    bsz, inp_len, vocab_sz = inp.shape
    
    if cls.vocab_weights is not None and cls.vocab_weights.shape[0] != vocab_sz: 
        raise ValueError(f"`vocab_weights` should have {vocab_sz} elements.")
    
    seq_len = min(targ_len, inp_len)
    inp, targ = -F.log_softmax(inp, dim=2)[:, :seq_len].transpose(1,2), targ[:, :seq_len]
    if cls.vocab_weights is not None: inp *= cls.vocab_weights.unsqueeze(1)
    
    if n_inp2targ is not None:
        mn_targ = n_inp2targ.max()
    
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
    else:
        if bsz != tn_targ: raise ValueError("`inp` and `targ` should have same number of elements as `n_inp2targ` is empty.")
        s = inp.gather(1, targ.view(bsz, -1, seq_len)).view(-1, seq_len); xtarg = targ
    
    idx = torch.where(xtarg != cls.ig_tok)
    loss = s[idx[0], idx[1]]
    
    if cls.reduction == 'mean': return (loss/len(torch.where(targ != cls.ig_tok)[0])).sum()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')


# %% ../nbs/04_losses.ipynb 36
class Calibration(BaseLoss):

    def __init__(self,
                 margin:Optional[float]=0.3,
                 tau:Optional[float]=0.1,
                 n_negatives:Optional[int]=10,
                 apply_softmax:Optional[bool]=True,
                 **kwargs):
        super().__init__(**kwargs)
        store_attr('margin,tau,n_negatives,apply_softmax')
        

# %% ../nbs/04_losses.ipynb 37
@patch
def forward(cls:Calibration,
            einp:torch.FloatTensor,
            inp:torch.FloatTensor, 
            targ:torch.LongTensor, 
            n_inp2targ:torch.LongTensor,
            inp2targ_idx:torch.LongTensor,
            n_pinp2targ:torch.LongTensor,
            pinp2targ_idx:torch.LongTensor,
            margin:Optional[float]=None,
            tau:Optional[float]=None,
            n_negatives:Optional[int]=None,
            apply_softmax:Optional[bool]=None,
            **kwargs):
    store_attr('margin', is_none=False)

    einp, inp, targ = einp.float(), inp.float(), targ.float()
    esc,sc = einp@targ.T,inp@targ.T
    
    _, idx = torch.unique(torch.cat([inp2targ_idx, pinp2targ_idx]), return_inverse=True)
    pos = get_sparse_matrix(idx[len(inp2targ_idx):], n_pinp2targ, size=(len(n_pinp2targ), idx.max()+1)).to_dense()[:, idx[:len(inp2targ_idx)]]

    mul = 2*pos - 1
    loss = F.relu((sc-esc)*mul + cls.margin)

    if cls.n_negatives is not None:
        loss, idx = torch.topk(loss, min(cls.n_negatives, loss.shape[1]), dim=1, largest=True)
        esc,sc,mul = esc.gather(1, idx), sc.gather(1, idx), mul.gather(1, idx)
    
    if cls.apply_softmax:
        m = loss != 0
        s = torch.where(mul == 1, sc, esc)
        p = s/cls.tau * m
        p = torch.softmax(p, dim=1)
        loss = loss*p
    
    if cls.reduction == 'mean': return loss.mean()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        

# %% ../nbs/04_losses.ipynb 56
class BaseMultiTriplet(BaseLoss):

    def __init__(
        self,
        margin:Optional[float]=0.8,
        tau:Optional[float]=0.1,
        apply_softmax:Optional[bool]=False,
        n_negatives:Optional[int]=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        store_attr('margin,tau,apply_softmax,n_negatives')

    def align_indices(self, indices:torch.Tensor, group_lengths:torch.Tensor):
        n, num_groups, max_len = len(indices), len(group_lengths), group_lengths.max()
        group_ids = torch.repeat_interleave(torch.arange(num_groups, device=indices.device), group_lengths)
    
        row_indices = torch.arange(n, device=indices.device)
    
        group_start = torch.cat([torch.zeros(1, dtype=group_lengths.dtype, device=group_lengths.device), group_lengths.cumsum(0)[:-1]], dim=0)
    
        within_idx = row_indices - group_start[group_ids]
    
        output = torch.zeros((num_groups, max_len), dtype=indices.dtype, device=indices.device)
        mask = torch.zeros((num_groups, max_len), device=indices.device)
        output[group_ids, within_idx] = indices
        mask[group_ids, within_idx] = 1.0
    
        return output, mask

    def remove_redundant_indices(self, inp2targ_idx:torch.Tensor, n_inp2targ:torch.Tensor, pinp2targ_idx:torch.Tensor, n_pinp2targ:torch.Tensor):
        mask = torch.isin(pinp2targ_idx, inp2targ_idx)
        new_pinp2targ_idx = pinp2targ_idx[mask]
    
        num_groups = len(n_pinp2targ)
        group_ids = torch.repeat_interleave(torch.arange(num_groups, device=n_pinp2targ.device), n_pinp2targ)
        new_n_pinp2targ = torch.bincount(group_ids[mask], minlength=num_groups)
    
        return new_pinp2targ_idx, new_n_pinp2targ

    def reset_indices(self, inp2targ_idx:torch.Tensor, n_inp2targ:torch.Tensor, pinp2targ_idx:torch.Tensor, n_pinp2targ:torch.Tensor):
        _, reset_indices, counts = torch.unique(torch.cat([inp2targ_idx, pinp2targ_idx]), return_inverse=True, return_counts=True)
    
        _, idx_sorted = torch.sort(reset_indices, stable=True)
        cum_sum = torch.cat((torch.zeros((1,), dtype=counts.dtype, device=counts.device), counts.cumsum(0)[:-1]))
        indices = idx_sorted[cum_sum]
    
        inp2targ_idx = reset_indices[:len(inp2targ_idx)]
        pinp2targ_idx = reset_indices[len(inp2targ_idx):]
    
        return inp2targ_idx, pinp2targ_idx, indices

    def compute_scores(self, inp, targ, indices=None):
        if indices is not None: targ = targ[indices]
        return inp@targ.T

    def forward(
        self, 
        
        n_inp2targ:torch.LongTensor,
        inp2targ_idx:torch.LongTensor,
        n_pinp2targ:torch.LongTensor,
        pinp2targ_idx:torch.LongTensor,

        inp:Optional[torch.FloatTensor]=None, 
        targ:Optional[torch.FloatTensor]=None,
        scores:Optional[torch.FloatTensor]=None,
        
        margin:Optional[float]=None,
        tau:Optional[float]=None,
        apply_softmax:Optional[bool]=None,
        n_negatives:Optional[int]=None,
        **kwargs
    ):
        store_attr('margin,tau,apply_softmax,n_negatives', is_none=False)

        inp, targ = inp.float(), targ.float()
        scores = scores if scores is None else scores.float()
        
        pinp2targ_idx, n_pinp2targ = self.remove_redundant_indices(inp2targ_idx, n_inp2targ, pinp2targ_idx, n_pinp2targ)
        inp2targ_idx, pinp2targ_idx, indices = self.reset_indices(inp2targ_idx, n_inp2targ, pinp2targ_idx, n_pinp2targ)

        scores = self.compute_scores(inp, targ, indices=indices) if scores is None else scores[:, indices]

        pos_indices, pos_mask = self.align_indices(inp2targ_idx, n_inp2targ)
        pos_scores = scores.gather(1, pos_indices)

        pos_incidence = torch.zeros_like(scores)
        
        ppos_indices, ppos_mask = self.align_indices(pinp2targ_idx, n_pinp2targ)
        pos_incidence = pos_incidence.scatter(1, ppos_indices, 1)

        ppos_indices[~ppos_mask.bool()] = -1
        row_idx = torch.where(torch.all(ppos_indices != 0, dim=1))[0]
        pos_incidence[row_idx, 0] = 0
        
        neg_incidence = 1 - pos_incidence

        loss = scores.unsqueeze(1) - pos_scores.unsqueeze(2) + self.margin
        loss = F.relu(loss * neg_incidence.unsqueeze(1))

        scores = scores.unsqueeze(1).expand_as(loss)
        neg_incidence = neg_incidence.unsqueeze(1).expand_as(loss)

        if self.n_negatives is not None:
            loss, idx = torch.topk(loss, min(self.n_negatives, loss.shape[2]), dim=2, largest=True)
            scores, neg_incidence = scores.gather(2, idx), neg_incidence.gather(2, idx)

        if self.apply_softmax:
            mask = loss != 0
            penalty = scores / self.tau * mask
            penalty[neg_incidence == 0] = torch.finfo(penalty.dtype).min
            penalty = torch.softmax(penalty, dim=2)
            loss = loss * penalty
            
        loss /= (neg_incidence.sum(dim=2, keepdim=True) + 1e-6)
        loss /= (n_inp2targ.unsqueeze(1).unsqueeze(1) + 1e-6)
        loss = loss[pos_mask.bool()].sum()
        
        if self.reduction == 'mean': return loss/len(n_inp2targ)
        elif self.reduction == 'sum': return loss
        else: raise ValueError(f'`reduction` cannot be `{self.reduction}`')
    

# %% ../nbs/04_losses.ipynb 58
class MultiTriplet(BaseMultiTriplet):

    def forward(
        self, 
        inp:torch.FloatTensor, # bs x dim
        targ:torch.FloatTensor, # total labels in batch (t) x dim
        n_inp2targ:torch.LongTensor, # bs x dim (like indptr in sp.csr_matrix)
        inp2targ_idx:torch.LongTensor, # t x dim (index of label)
        n_pinp2targ:torch.LongTensor,
        pinp2targ_idx:torch.LongTensor,
        margin:Optional[float]=None,
        tau:Optional[float]=None,
        apply_softmax:Optional[bool]=None,
        n_negatives:Optional[int]=None,
        **kwargs
    ):
        return super().forward(n_inp2targ, inp2targ_idx, n_pinp2targ, pinp2targ_idx, inp=inp, targ=targ, margin=margin, tau=tau, 
                               apply_softmax=apply_softmax, n_negatives=n_negatives, **kwargs)
        

# %% ../nbs/04_losses.ipynb 59
class MultiTripletFromInBatchScores(BaseMultiTriplet):

    def forward(
        self, 
        scores:torch.FloatTensor,  
        n_inp2targ:torch.LongTensor,
        inp2targ_idx:torch.LongTensor,
        n_pinp2targ:torch.LongTensor,
        pinp2targ_idx:torch.LongTensor,
        margin:Optional[float]=None,
        tau:Optional[float]=None,
        apply_softmax:Optional[bool]=None,
        n_negatives:Optional[int]=None,
        **kwargs
    ):
        return super().forward(n_inp2targ, inp2targ_idx, n_pinp2targ, pinp2targ_idx, scores=scores, margin=margin, tau=tau, 
                               apply_softmax=apply_softmax, n_negatives=n_negatives, **kwargs)
        

# %% ../nbs/04_losses.ipynb 72
class MultiTripletFromScores(BaseMultiTriplet):

    @staticmethod
    def get_incidence(n_inp2targ:int, inp2targ_idx:torch.Tensor, n_pinp2targ:torch.Tensor, pinp2targ_idx:torch.Tensor):
        row_idx = torch.arange(len(n_pinp2targ), device=n_pinp2targ.device)
        inp2targ_row_idx = row_idx.repeat_interleave(n_inp2targ)
        pinp2targ_row_idx = row_idx.repeat_interleave(n_pinp2targ)
        
        max_col_idx = max(int(inp2targ_idx.max()), int(pinp2targ_idx.max()))
        offset = max_col_idx + 1
    
        inp2targ_keys = inp2targ_row_idx * offset + inp2targ_idx
        pinp2targ_keys = pinp2targ_row_idx * offset + pinp2targ_idx
    
        return torch.isin(inp2targ_keys, pinp2targ_keys)

    @staticmethod
    def get_pos_scores(scores:torch.FloatTensor, n_inp2targ:torch.FloatTensor):
        row_idx = torch.arange(len(n_inp2targ), device=n_inp2targ.device)
        inp2targ_row_idx = row_idx.repeat_interleave(n_inp2targ)
        inp2targ_col_idx = torch.arange(n_inp2targ.sum(), device=n_inp2targ.device)
        return scores[inp2targ_row_idx, inp2targ_col_idx]
    
    def forward(
        self, 
        scores:torch.FloatTensor,  
        inp2targ_idx:torch.LongTensor,
        
        n_inp2targ:torch.LongTensor,
        
        n_pinp2targ:torch.LongTensor,
        pinp2targ_idx:torch.LongTensor,
        
        margin:Optional[float]=None,
        tau:Optional[float]=None,
        apply_softmax:Optional[bool]=None,
        n_negatives:Optional[int]=None,
        **kwargs
    ):
        store_attr('margin,tau,apply_softmax,n_negatives', is_none=False)
        
        assert scores.dim() == 2, "`scores` should be two dimensional matrix."
        assert inp2targ_idx.dim() == 2, "`inp2targ_idx` should be two dimensional matrix."
        
        pos_incidence = self.get_incidence(inp2targ_idx.shape[1], inp2targ_idx.flatten(), n_pinp2targ, pinp2targ_idx)
        pos_incidence = pos_incidence.view(inp2targ_idx.shape)

        pos_scores = self.get_pos_scores(scores, n_inp2targ)
        pos_scores, pos_mask = self.align_indices(pos_scores, n_inp2targ)
        neg_incidence = ~pos_incidence

        loss = scores.unsqueeze(1) - pos_scores.unsqueeze(2) + self.margin
        loss = F.relu(loss * neg_incidence.unsqueeze(1))
        
        scores = scores.unsqueeze(1).expand_as(loss)
        neg_incidence = neg_incidence.unsqueeze(1).expand_as(loss)

        if self.n_negatives is not None:
            loss, idx = torch.topk(loss, min(self.n_negatives, loss.shape[2]), dim=2, largest=True)
            scores, neg_incidence = scores.gather(2, idx), neg_incidence.gather(2, idx)

        if self.apply_softmax:
            mask = loss != 0
            penalty = scores / self.tau * mask
            penalty[neg_incidence == 0] = torch.finfo(penalty.dtype).min
            penalty = torch.softmax(penalty, dim=2)
            loss = loss * penalty

        loss /= (neg_incidence.sum(dim=2, keepdim=True) + 1e-6)
        loss /= (n_inp2targ.unsqueeze(1).unsqueeze(1) + 1e-6)
        loss = loss[pos_mask.bool()].sum()
        
        if self.reduction == 'mean': return loss/len(n_inp2targ)
        elif self.reduction == 'sum': return loss
        else: raise ValueError(f'`reduction` cannot be `{self.reduction}`')
        

# %% ../nbs/04_losses.ipynb 74
class MultiTripletWithNegatives(MultiTripletFromScores):

    def get_scores(self, inp:torch.Tensor, pos_targ:torch.Tensor, neg_targ:Optional[torch.Tensor]=None, n_neg:Optional[int]=None):
        scores = inp @ pos_targ.T
        if neg_targ is not None:
            neg_scores = inp.unsqueeze(1) @ neg_targ.view(len(inp), n_neg, -1).transpose(1, 2)
            neg_scores = neg_scores.squeeze(1)
            scores = torch.hstack([scores, neg_scores])
        return scores

    def get_indices(self, pos_idx:torch.Tensor, bsz:int, neg_idx:Optional[torch.Tensor]=None, n_neg:Optional[int]=None):
        indices = torch.repeat_interleave(pos_idx.unsqueeze(0), bsz, 0)
        if neg_idx is not None:
            neg_idx = neg_idx.view(bsz, n_neg)
            indices = torch.hstack([indices, neg_idx])
        return indices

    def forward(
        self, 
        inp:torch.FloatTensor,
        
        pos_targ:torch.FloatTensor,
        n_pos:torch.LongTensor,
        pos_idx:torch.LongTensor,

        neg_targ:torch.FloatTensor,
        n_neg:torch.LongTensor,
        neg_idx:torch.LongTensor,
        
        n_ppos:torch.LongTensor,
        ppos_idx:torch.LongTensor,
        
        margin:Optional[float]=None,
        tau:Optional[float]=None,
        apply_softmax:Optional[bool]=None,
        n_negatives:Optional[int]=None,
        **kwargs
    ):  
        assert torch.all(n_neg == n_neg.max()), "All datapoints should same number of negatives"
        scores = self.get_scores(inp, pos_targ, neg_targ, n_neg.max())
        indices = self.get_indices(pos_idx, len(inp), neg_idx, n_neg.max())
        return super().forward(scores, indices, n_pos, n_pinp2targ=n_ppos, pinp2targ_idx=ppos_idx)
    

# %% ../nbs/04_losses.ipynb 89
class MarginMSEWithNegatives(BaseLoss):

    def forward(
        self, 
        inp:torch.FloatTensor,
        
        pos_targ:torch.FloatTensor,
        pos_scores:torch.FloatTensor,

        neg_targ:torch.FloatTensor,
        neg_scores:torch.FloatTensor,
        **kwargs
    ):  
        bsz = len(inp)
        
        assert len(pos_scores) % bsz == 0, "Number of elements in `pos_scores` should be divisible by batch size."
        assert len(neg_scores) % bsz == 0, "Number of elements in `neg_scores` should be divisible by batch size."
        
        assert len(pos_targ) == len(pos_scores), "`pos_targ` and `pos_scores` should have same number of elements."
        assert len(neg_targ) == len(neg_scores), "`neg_targ` and `neg_scores` should have same number of elements."
        
        n = len(pos_targ) // bsz
        pos_targ, pos_scores = pos_targ.view(bsz, n, -1), pos_scores.view(bsz, n, 1)
        n = len(neg_targ) // bsz
        neg_targ, neg_scores = neg_targ.view(bsz, n, -1), neg_scores.view(bsz, 1, n)

        labels = pos_scores - neg_scores
        
        inp = inp.unsqueeze(1)
        pos_scores = inp @ pos_targ.transpose(1, 2)
        neg_scores = inp @ neg_targ.transpose(1, 2)
        margins = pos_scores.transpose(1, 2) - neg_scores

        return F.mse_loss(margins.flatten(), labels.flatten())
        

# %% ../nbs/04_losses.ipynb 104
class Cosine(BaseLoss):

    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
        

# %% ../nbs/04_losses.ipynb 105
@patch
def forward(cls:Cosine, 
            inp:torch.FloatTensor,
            inp_mask:torch.FloatTensor,
            targ:torch.LongTensor,
            targ_mask:torch.LongTensor,
            **kwargs):
    seq_len = min(inp.shape[1], targ.shape[1])
    
    inp_mask = inp_mask.unsqueeze(2).expand(inp.size()).float()
    targ_mask = targ_mask.unsqueeze(2).expand(targ.size()).float()

    inp, targ = F.normalize(inp, dim=-1),F.normalize(targ, dim=-1)
    
    inp,targ = inp*inp_mask,targ*targ_mask
    inp,targ = inp[:,:seq_len],targ[:,:seq_len]

    loss = 1.0 - torch.sum(inp*targ, dim=-1)
    
    if cls.reduction == 'mean': return loss.mean()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        

# %% ../nbs/04_losses.ipynb 111
class Entropy(BaseLoss):

    def __init__(self, 
                 margin:Optional[float]=0.8,
                 tau:Optional[float]=0.1,
                 apply_softmax:Optional[bool]=True,
                 n_negatives:Optional[int]=5,
                 **kwargs):
        super().__init__(**kwargs)
        store_attr('margin,tau,apply_softmax,n_negatives')
        

# %% ../nbs/04_losses.ipynb 112
@patch
def forward(cls:Entropy, 
            inp:torch.FloatTensor,
            targ:torch.LongTensor,
            inp2targ_idx:torch.LongTensor,
            n_pinp2targ:torch.LongTensor,
            pinp2targ_idx:torch.LongTensor,
            margin:Optional[float]=None,
            tau:Optional[float]=None,
            apply_softmax:Optional[bool]=None,
            n_negatives:Optional[int]=None,
            **kwargs):
    store_attr('margin,tau,apply_softmax,n_negatives', is_none=False)
    _, idx = torch.unique(torch.cat([inp2targ_idx, pinp2targ_idx]), return_inverse=True)
    ne = 1 - get_sparse_matrix(idx[len(inp2targ_idx):], n_pinp2targ).to_dense()[:, idx[:len(inp2targ_idx)]]
    
    sc = targ.exp()@inp.T
    
    sc_p =  sc.diagonal().unsqueeze(1)
    _, ne_idx = torch.topk(torch.where(ne == 0, torch.finfo(sc.dtype).min, sc), min(cls.n_negatives, sc.shape[0]-1), dim=1, largest=True)
    sc_n = sc.gather(1, ne_idx)
    
    loss = torch.relu(sc_n - sc_p + cls.margin)
    
    if cls.apply_softmax:
        m = loss != 0
        p = torch.softmax(sc_n/cls.tau * m, dim=1)
        loss = loss*p
    
    if cls.reduction == 'mean': return loss.mean()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        

# %% ../nbs/04_losses.ipynb 118
class Triplet(BaseLoss):

    def __init__(self, 
                 margin:Optional[float]=0.8,
                 tau:Optional[float]=0.1,
                 apply_softmax:Optional[bool]=True,
                 n_negatives:Optional[int]=5,
                 **kwargs):
        super().__init__(**kwargs)
        store_attr('margin,tau,apply_softmax,n_negatives')


# %% ../nbs/04_losses.ipynb 119
@patch
def forward(cls:Triplet, 
            inp:torch.FloatTensor, 
            targ:torch.LongTensor, 
            inp2targ_idx:torch.LongTensor,
            n_pinp2targ:torch.LongTensor,
            pinp2targ_idx:torch.LongTensor,
            margin:Optional[float]=None,
            tau:Optional[float]=None,
            apply_softmax:Optional[bool]=None,
            n_negatives:Optional[int]=None,
            **kwargs):
    store_attr('margin,tau,apply_softmax,n_negatives', is_none=False)
    _, idx = torch.unique(torch.cat([inp2targ_idx, pinp2targ_idx]), return_inverse=True)
    ne = 1 - get_sparse_matrix(idx[len(inp2targ_idx):], n_pinp2targ).to_dense()[:, idx[:len(inp2targ_idx)]]
    
    sc = inp@targ.T
    sc_p =  sc.diagonal().unsqueeze(1)
    _, ne_idx = torch.topk(torch.where(ne == 0, -10, sc), min(cls.n_negatives, sc.shape[0]-1), dim=1, largest=True)
    sc_n = sc.gather(1, ne_idx)
    
    loss = torch.relu(sc_n - sc_p + cls.margin)
    
    if cls.apply_softmax:
        m = loss != 0
        p = torch.softmax(sc_n/cls.tau * m, dim=1)
        loss = loss*p
    
    if cls.reduction == 'mean': return loss.mean()
    elif cls.reduction == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        

# %% ../nbs/04_losses.ipynb 124
class SoupCon(BaseLoss):

    @delegates(BaseLoss.__init__)
    def __init__(self,
                 tau:Optional[float]=1.0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        

# %% ../nbs/04_losses.ipynb 126
@patch
def forward(cls:SoupCon,
            inp:torch.FloatTensor,
            targ:torch.LongTensor,
            n_inp2targ:torch.LongTensor,
            inp2targ_idx:torch.LongTensor,
            **kwargs):
    _, idx, cnt = torch.unique(inp2targ_idx, return_inverse=True, return_counts=True)
    _, idx_sorted = torch.sort(idx)
    targ_idx = idx_sorted[torch.cat([torch.zeros(1, dtype=inp2targ_idx.dtype, device=inp2targ_idx.device), cnt.cumsum(0)[:-1]])]

    pe = get_sparse_matrix(idx, n_inp2targ).to_dense()
    sc = -F.log_softmax(inp@targ[targ_idx].T/cls.tau, dim=1)
    
    loss = sc*pe
    loss /= pe.sum(dim=1, keepdim=True)
    
    if cls.reduce == 'mean': return loss.sum(dim=1).mean()
    elif cls.reduce == 'sum': return loss.sum()
    else: raise ValueError(f'`reduction` cannot be `{cls.reduction}`')
        
