# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/27_models.classifiers.ipynb.

# %% auto 0
__all__ = ['CLS001']

# %% ../../nbs/27_models.classifiers.ipynb 2
import torch, numpy as np
from typing import Optional
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

from ..core import store_attr
from ..losses import MultiTriplet
from .modeling_utils import XCModelOutput

from transformers.utils.generic import ModelOutput
from transformers import DistilBertPreTrainedModel,DistilBertConfig

# %% ../../nbs/27_models.classifiers.ipynb 15
class CLS001(DistilBertPreTrainedModel):
    use_generation,use_representation, = False,True
    
    def __init__(
        self, 
        config, 
        n_data:int, 
        n_lbl:int, 
        num_batch_labels:Optional[int]=None, 
        batch_size:Optional[int]=None,
        margin:Optional[float]=0.3,
        num_negatives:Optional[int]=5,
        tau:Optional[float]=0.1,
        apply_softmax:Optional[bool]=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        store_attr('n_data,n_lbl')
        self.data_repr = nn.Embedding(self.n_data, config.dim)
        self.lbl_repr = nn.Embedding(self.n_lbl, config.dim)
        self.lbl_embeddings = nn.Embedding(self.n_lbl, config.dim)

        self.rep_loss_fn = MultiTriplet(bsz=batch_size, tn_targ=num_batch_labels, margin=margin, n_negatives=num_negatives, 
                                        tau=tau, apply_softmax=apply_softmax, reduce='mean')

    def get_lbl_representation(self):
        return self.lbl_repr.weight

    def get_data_representation(self):
        return self.data_repr.weight

    def init_representation(self, data_repr:torch.Tensor, lbl_repr:torch.Tensor):
        self.data_repr.weight.data = data_repr
        self.lbl_repr.weight.data = lbl_repr

    def freeze_representation(self):
        self.data_repr.requires_grad_(False)
        self.lbl_repr.requires_grad_(False)

    def freeze_data_representation(self):
        self.data_repr.requires_grad_(False)

    def unfreeze_representation(self):
        self.data_repr.requires_grad_(True)
        self.lbl_repr.requires_grad_(True)

    def init_meta_embeddings(self):
        self.lbl_embeddings.weight.data = torch.zeros_like(self.lbl_embeddings.weight.data)

    def compute_loss(self, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx):
        return self.rep_loss_fn(inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx)

    def forward(
        self,
        data_idx:torch.Tensor,
        lbl2data_idx:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        plbl2data_idx:Optional[torch.Tensor]=None,
        plbl2data_data2ptr:Optional[torch.Tensor]=None,
    ):
        data_rep = self.data_repr(data_idx)

        loss = lbl2data_rep = None
        if lbl2data_idx is not None:
            lbl2data_rep = F.normalize(self.lbl_repr(lbl2data_idx) + self.lbl_embeddings(lbl2data_idx))

            loss = self.compute_loss(data_rep, lbl2data_rep,lbl2data_data2ptr,lbl2data_idx,
                                     plbl2data_data2ptr,plbl2data_idx)
            
        return XCModelOutput(
            loss=loss,
            data_repr=data_rep,
            lbl2data_repr=lbl2data_rep,
        )
        
        
