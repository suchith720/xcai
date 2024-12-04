# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/34_models.NVM0XX.ipynb.

# %% auto 0
__all__ = ['Pooling', 'RepresentationHead', 'NVM009Encoder', 'NVM009']

# %% ../../nbs/34_models.NVM0XX.ipynb 2
import torch, re, inspect, pickle, os, torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Mapping, Any, Union

from transformers.activations import get_activation

from fastcore.meta import *

from ..core import store_attr
from ..losses import MultiTriplet
from .modeling_nvembed import NVEmbedModel
from .modeling_utils import XCModelOutput, Pooling

# %% ../../nbs/34_models.NVM0XX.ipynb 10
class Pooling:

    @staticmethod
    def mean_pooling(data_embeds:torch.FloatTensor, data_attention_mask:torch.LongTensor):
        data_attention_mask = data_attention_mask.unsqueeze(2).expand(data_embeds.size()).float()
        return torch.sum(data_embeds * data_attention_mask, 1) / torch.clamp(data_attention_mask.sum(1), min=1e-9)

    @staticmethod
    def last_pooling(data_embeds:torch.FloatTensor, data_attention_mask:torch.LongTensor):
        index = data_attention_mask.sum(dim=1) - 1
        return data_embeds[torch.arange(data_embeds.size(0), device=data_embeds.device), index]
    

# %% ../../nbs/34_models.NVM0XX.ipynb 11
class RepresentationHead(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation('relu')
        
        self.post_init()
        
    def post_init(self):
        torch.nn.init.eye_(self.transform.weight)
        torch.nn.init.eye_(self.projector.weight)
        
        torch.nn.init.zeros_(self.transform.bias)
        torch.nn.init.zeros_(self.projector.bias)
        
    def forward(self, x:torch.Tensor):
        x = self.transform(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.projector(x)
        return x
    

# %% ../../nbs/34_models.NVM0XX.ipynb 13
class NVM009Encoder(NVEmbedModel):
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.dr_head = RepresentationHead(config)
        
    @delegates(NVEmbedModel.__call__)
    def forward(
        self, 
        input_ids:Optional[torch.Tensor]=None, 
        attention_mask:Optional[torch.Tensor]=None,
        pool_mask: Optional[torch.Tensor]=None,
        return_dict: bool=True,
        **kwargs
    ):
        outputs = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        embeds = self.latent_attention_model(
            outputs.last_hidden_state,
            pool_mask,
        )
        rep = self.dr_head(embeds)
        
        return outputs, F.normalize(Pooling.mean_pooling(rep, attention_mask), dim=1)
    

# %% ../../nbs/34_models.NVM0XX.ipynb 14
class NVM009(NVEmbedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.embedding_model,encoder.latent_attention_model"]
    
    def __init__(self,
                 config,
                 bsz:Optional[int]=None,
                 tn_targ:Optional[int]=None,
                 margin:Optional[float]=0.3,
                 tau:Optional[float]=0.1,
                 apply_softmax:Optional[bool]=False,
                 n_negatives:Optional[int]=5,
                 use_encoder_parallel:Optional[bool]=True,
                 *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        store_attr('use_encoder_parallel')
        self.encoder = NVM009Encoder(config)
        self.loss_fn = MultiTriplet(bsz=bsz, tn_targ=tn_targ, margin=margin, n_negatives=n_negatives, tau=tau, 
                                    apply_softmax=apply_softmax, reduce='mean')
        self.post_init()
        self.remap_post_init()
        
    def init_dr_head(self):
        self.encoder.dr_head.post_init()
        
    def remap_post_init(self):
        self.embedding_model = self.encoder.embedding_model
        self.latent_attention_model = self.encoder.latent_attention_model
    
    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_idx:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        plbl2data_data2ptr:Optional[torch.Tensor]=None,
        plbl2data_idx:Optional[torch.Tensor]=None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.use_encoder_parallel: 
            encoder = nn.DataParallel(module=self.encoder)
        else: encoder = self.encoder
        
        data_o, data_repr = encoder(data_input_ids, data_attention_mask)
        
        loss, lbl2data_repr = None, None
        if lbl2data_input_ids is not None:
            lbl2data_o, lbl2data_repr = encoder(lbl2data_input_ids, lbl2data_attention_mask)
            
            loss = self.loss_fn(data_repr, lbl2data_repr, lbl2data_data2ptr, lbl2data_idx, 
                                plbl2data_data2ptr, plbl2data_idx, **kwargs)

        if not return_dict:
            o = (data_repr, lbl2data_repr)
            return ((loss,) + o) if loss is not None else o

        return XCModelOutput(
            loss=loss,
            data_repr=data_repr,
            lbl2data_repr=lbl2data_repr,
        )