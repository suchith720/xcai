# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/20_models.oak.ipynb.

# %% auto 0
__all__ = ['CrossAttention', 'NormCrossAttention', 'Encoder', 'OAK000', 'OAK001', 'Encoder002', 'OAK002', 'Encoder003', 'OAK003']

# %% ../../nbs/20_models.oak.ipynb 2
import torch, re, inspect, pickle, os, torch.nn as nn, math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Mapping, Any, Union
from transformers import (
    PretrainedConfig,
    DistilBertForMaskedLM,
    DistilBertModel,
    DistilBertPreTrainedModel,
)
from transformers.utils.generic import ModelOutput
from transformers.activations import get_activation

from fastcore.meta import *
from fastcore.utils import *

from ..losses import *
from ..core import store_attr
from ..learner import XCDataParallel
from .modeling_utils import *

# %% ../../nbs/20_models.oak.ipynb 18
class CrossAttention(nn.Module):
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config, self.n_h, self.dim = config, config.n_heads, config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        if self.dim % self.n_h != 0:
            raise ValueError(f"self.n_heads: {self.n_h} must divide self.dim: {self.dim} evenly.")
            
        self.q = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.o = nn.Linear(in_features=config.dim, out_features=config.dim)

    def post_init(self):
        self.q.weight.data = torch.eye(self.q.out_features, self.q.in_features, dtype=self.q.weight.dtype)
        self.k.weight.data = torch.eye(self.k.out_features, self.k.in_features, dtype=self.k.weight.dtype)
        self.v.weight.data = torch.eye(self.v.out_features, self.v.in_features, dtype=self.v.weight.dtype)
        self.o.weight.data = torch.eye(self.o.out_features, self.o.in_features, dtype=self.o.weight.dtype)

    def forward(
        self, 
        q: torch.Tensor,
        q_m: torch.Tensor,
        k: torch.Tensor, 
        k_m: torch.Tensor,
        output_attentions:Optional[bool] = False,
    ):
        bs, q_len, dim = q.size()
        v, k_len = k, k.size(1) 

        h_dim = self.dim//self.n_h

        def shape(x: torch.Tensor): return x.view(bs, -1, self.n_h, h_dim).transpose(1, 2)

        def unshape(x: torch.Tensor): return x.transpose(1, 2).contiguous().view(bs, -1, self.n_h * h_dim)

        q = shape(self.q(q))  # (bs, n_h, q_len, h_dim)
        k = shape(self.k(k))  # (bs, n_h, k_len, h_dim)
        v = shape(self.v(v))  # (bs, n_h, k_len, h_dim)

        q = q / math.sqrt(h_dim)  # (bs, n_h, q_len, h_dim)
        sc = torch.matmul(q, k.transpose(2, 3))  # (bs, n_h, q_len, k_len)
        
        q_m, k_m = q_m.view(bs, 1, -1, 1).to(q.dtype), k_m.view(bs, 1, 1, -1).to(q.dtype)
        mask = torch.matmul(q_m, k_m).expand_as(sc)  # (bs, n_h, q_len, k_len)
        
        sc = sc.masked_fill(mask == 0, torch.tensor(torch.finfo(sc.dtype).min))  # (bs, n_h, q_len, k_len)

        w = nn.functional.softmax(sc, dim=-1)  # (bs, n_h, q_len, k_len)
        w = self.dropout(w)  # (bs, n_h, q_len, k_len)

        o = self.o(unshape(torch.matmul(w, v))) # (bs, q_len, dim)
        
        if output_attentions: return (o, w)
        else: return (o,)
        

# %% ../../nbs/20_models.oak.ipynb 25
class NormCrossAttention(nn.Module):
    
    def __init__(self, config: PretrainedConfig, tau:Optional[float]=0.1, dropout:Optional[float]=0.1):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        
        self.config, self.n_h, self.dim = config, config.n_heads, config.dim
        self.dropout = nn.Dropout(p=dropout)

        if self.dim % self.n_h != 0:
            raise ValueError(f"self.n_heads: {self.n_h} must divide self.dim: {self.dim} evenly.")
            
        self.q = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.o = nn.Linear(in_features=config.dim, out_features=config.dim)

    def post_init(self):
        self.q.weight.data = torch.eye(self.q.out_features, self.q.in_features, dtype=self.q.weight.dtype)
        self.k.weight.data = torch.eye(self.k.out_features, self.k.in_features, dtype=self.k.weight.dtype)
        self.v.weight.data = torch.eye(self.v.out_features, self.v.in_features, dtype=self.v.weight.dtype)
        self.o.weight.data = torch.eye(self.o.out_features, self.o.in_features, dtype=self.o.weight.dtype)

    def forward(
        self, 
        q: torch.Tensor,
        q_m: torch.Tensor,
        k: torch.Tensor, 
        k_m: torch.Tensor,
        output_attentions:Optional[bool] = False,
    ):
        bs, q_len, dim = q.size()
        v, k_len = k, k.size(1) 

        h_dim = self.dim//self.n_h

        def shape(x: torch.Tensor): return x.view(bs, -1, self.n_h, h_dim).transpose(1, 2)

        def unshape(x: torch.Tensor): return x.transpose(1, 2).contiguous().view(bs, -1, self.n_h * h_dim)

        q = shape(self.q(q))  # (bs, n_h, q_len, h_dim)
        k = shape(self.k(k))  # (bs, n_h, k_len, h_dim)
        v = shape(self.v(v))  # (bs, n_h, k_len, h_dim)

        q = q / math.sqrt(h_dim)  # (bs, n_h, q_len, h_dim)
        sc = torch.matmul(q, k.transpose(2, 3))  # (bs, n_h, q_len, k_len)
        sc = sc / self.tau
        
        q_m, k_m = q_m.view(bs, 1, -1, 1).to(q.dtype), k_m.view(bs, 1, 1, -1).to(q.dtype)
        mask = torch.matmul(q_m, k_m).expand_as(sc)  # (bs, n_h, q_len, k_len)
        
        sc = sc.masked_fill(mask == 0, torch.tensor(torch.finfo(sc.dtype).min))  # (bs, n_h, q_len, k_len)
        
        w = nn.functional.softmax(sc, dim=-1)  # (bs, n_h, q_len, k_len)
        w = self.dropout(w)  # (bs, n_h, q_len, k_len)

        o = self.o(unshape(torch.matmul(w, v))) # (bs, q_len, dim)
        
        if output_attentions: return (o, w)
        else: return (o,)
        

# %% ../../nbs/20_models.oak.ipynb 32
class Encoder(DistilBertPreTrainedModel):
    
    def __init__(
        self, 
        config:PretrainedConfig, 
        num_metadata:int,
        resize_length:Optional[int]=None,
    ):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        
        self.dr_head = RepresentationHead(config)
        self.dr_fused_head = RepresentationHead(config)
        self.meta_head = RepresentationHead(config)
        self.cross_head = CrossAttention(config)
        self.meta_embeddings = nn.Embedding(num_metadata, config.dim)

        self.ones = torch.ones(resize_length, dtype=torch.long, device=self.device) if resize_length is not None else None
        self.post_init()

    def freeze_meta_embeddings(self):
        self.meta_embeddings.requires_grad_(False)

    def unfreeze_meta_embeddings(self):
        self.meta_embeddings.requires_grad_(True)

    def set_meta_embeddings(self, embed:torch.Tensor):
        self.meta_embeddings.weight.data = embed
        
    def get_position_embeddings(self) -> nn.Embedding:
        return self.distilbert.get_position_embeddings()
    
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)
    
    def encode(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, **kwargs):
        return self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def dr(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.dr_head(embed)
        return F.normalize(Pooling.mean_pooling(embed, attention_mask), dim=1)

    def dr_fused(self, embed:torch.Tensor):
        embed = self.dr_fused_head(embed)
        return F.normalize(embed, dim=1)

    def meta(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.meta_head(embed)
        return F.normalize(Pooling.mean_pooling(embed, attention_mask), dim=1)
    
    def meta_unnormalized(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.meta_head(embed)
        return Pooling.mean_pooling(embed, attention_mask)

    def resize(self, idx:torch.Tensor, num_inputs:torch.Tensor):
        if torch.any(num_inputs == 0): raise ValueError("`num_inputs` should be non-zero positive integer.")
        bsz, total_num_inputs = num_inputs.shape[0], idx.shape[0]
        
        self.ones = self.ones.to(idx.device)
        ones = (
            torch.ones(total_num_inputs, dtype=torch.long, device=idx.device) 
            if self.ones is None or self.ones.shape[0] < total_num_inputs else self.ones[:total_num_inputs]
        )

        max_num_inputs = num_inputs.max()
        xnum_inputs = max_num_inputs-num_inputs+1

        inputs_ptr = num_inputs.cumsum(dim=0)-1
        repeat_inputs = ones.scatter(0, inputs_ptr, xnum_inputs)
        
        resized_idx = idx.repeat_interleave(repeat_inputs, dim=0)
        ignore_mask = ones.scatter(0, inputs_ptr, 0).repeat_interleave(repeat_inputs, dim=0).view(bsz, -1)
        ignore_mask[:, -1] = 1; ignore_mask = ignore_mask.flatten()
        
        return resized_idx,ignore_mask

    
    def fuse_meta_into_embeddings(self, data_repr:torch.Tensor, data_mask:torch.Tensor, meta_kwargs:Dict):
        meta_repr = {}
        
        data_fused_repr, data_mask = data_repr.clone().view(-1, 1, self.config.dim), data_mask.view(-1, 1)
        for m_key, m_args in meta_kwargs.items():
            idx = torch.where(m_args['data2ptr'] > 0)[0]
            meta_repr[m_key] = torch.empty(0, self.config.dim).to(data_repr)
            
            if len(idx):
                m_idx,m_repr_mask = self.resize(m_args['idx'], m_args['data2ptr'][idx])
                m_repr = F.normalize(self.meta_embeddings(m_idx), dim=1)
                
                m_repr, m_repr_mask = m_repr.view(len(idx), -1, self.config.dim), m_repr_mask.bool().view(len(idx), -1)
                meta_repr[m_key] = m_repr[m_repr_mask]
                
                fused_repr = self.cross_head(data_fused_repr[idx], data_mask[idx], m_repr, m_repr_mask)[0]
                data_fused_repr[idx] += fused_repr
                
        return data_fused_repr.squeeze(), meta_repr

    def forward(
        self, 
        data_input_ids: torch.Tensor, 
        data_attention_mask: torch.Tensor,
        data_aug_meta_prefix: Optional[str]=None,
        data_type:Optional[str]=None,
        data_unnormalized:Optional[bool]=False,
        **kwargs
    ):  
        data_o = self.encode(data_input_ids, data_attention_mask)
        
        if data_type is not None and data_type == "meta":
            data_repr = self.meta_unnormalized(data_o[0], data_attention_mask) if data_unnormalized else self.meta(data_o[0], data_attention_mask)
        else: 
            data_repr = self.dr(data_o[0], data_attention_mask)
        
        data_fused_repr = meta_repr = None
        if data_aug_meta_prefix is not None:
            meta_kwargs = Parameters.from_meta_aug_prefix(data_aug_meta_prefix, **kwargs)
            if len(meta_kwargs):
                data_fused_repr, meta_repr = self.fuse_meta_into_embeddings(data_repr, 
                                                                            torch.any(data_attention_mask, dim=1), 
                                                                            meta_kwargs)
                data_fused_repr = self.dr_fused(data_fused_repr)
                
        return EncoderOutput(
            rep=data_repr,
            fused_rep=data_fused_repr,
            meta_repr=meta_repr,
        )
        

# %% ../../nbs/20_models.oak.ipynb 34
class OAK000(nn.Module):
    
    def __init__(
        self, config,

        data_aug_meta_prefix:Optional[str]=None, 
        lbl2data_aug_meta_prefix:Optional[str]=None, 

        data_pred_meta_prefix:Optional[str]=None,
        lbl2data_pred_meta_prefix:Optional[str]=None,
        
        num_batch_labels:Optional[int]=None, 
        batch_size:Optional[int]=None,
        margin:Optional[float]=0.3,
        num_negatives:Optional[int]=5,
        tau:Optional[float]=0.1,
        apply_softmax:Optional[bool]=True,

        calib_margin:Optional[float]=0.3,
        calib_num_negatives:Optional[int]=10,
        calib_tau:Optional[float]=0.1,
        calib_apply_softmax:Optional[bool]=False,
        calib_loss_weight:Optional[float]=0.1,
        use_calib_loss:Optional[float]=False,
        
        meta_loss_weight:Optional[Union[List,float]]=0.3,
        
        use_fusion_loss:Optional[bool]=False,
        fusion_loss_weight:Optional[float]=0.15,

        use_query_loss:Optional[float]=True,
        
        use_encoder_parallel:Optional[bool]=True,
    ):
        super().__init__(config)
        store_attr('meta_loss_weight,fusion_loss_weight,calib_loss_weight')
        store_attr('data_pred_meta_prefix,lbl2data_pred_meta_prefix')
        store_attr('data_aug_meta_prefix,lbl2data_aug_meta_prefix')
        store_attr('use_fusion_loss,use_query_loss,use_calib_loss,use_encoder_parallel')
        
        self.encoder = None
        self.rep_loss_fn = MultiTriplet(bsz=batch_size, tn_targ=num_batch_labels, margin=margin, n_negatives=num_negatives, 
                                        tau=tau, apply_softmax=apply_softmax, reduce='mean')
        self.cab_loss_fn = Calibration(margin=calib_margin, tau=calib_tau, n_negatives=calib_num_negatives, 
                                       apply_softmax=calib_apply_softmax, reduce='mean')
        
    def init_retrieval_head(self):
        if self.encoder is None: raise ValueError('`self.encoder` is not initialized.')
        self.encoder.dr_head.post_init()
        self.encoder.meta_head.post_init()
        self.encoder.dr_fused_head.post_init()

    def init_cross_head(self):
        if self.encoder is None: raise ValueError('`self.encoder` is not initialized.')
        self.encoder.cross_head.post_init()
        

    def compute_loss(self, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx):
        return self.rep_loss_fn(inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx)

    def calibration_loss(self, einp_repr, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx):
        return self.calib_loss_weight * self.cab_loss_fn(einp_repr, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx)
    
    def compute_meta_loss(self, data_repr, lbl2data_repr, **kwargs):
        if self.use_encoder_parallel: 
            encoder = XCDataParallel(module=self.encoder)
        else: encoder = self.encoder
            
        data_meta_inputs = Parameters.from_meta_pred_prefix(self.data_pred_meta_prefix, **kwargs)
        lbl2data_meta_inputs = Parameters.from_meta_pred_prefix(self.lbl2data_pred_meta_prefix, **kwargs)
        meta_inputs = {**data_meta_inputs, **lbl2data_meta_inputs}

        m_lw = Parameters.get_meta_loss_weights(self.meta_loss_weight, len(meta_inputs)) if len(meta_inputs) else []
        
        loss = 0.0
        for inputs,lw in zip(meta_inputs.values(), m_lw):
            if 'lbl2data2ptr' in inputs:
                idx = torch.where(inputs['lbl2data2ptr'])[0]
                if len(idx) > 0:
                    inputs_o = encoder(data_input_ids=inputs['input_ids'], data_attention_mask=inputs['attention_mask'], 
                                       data_type="meta")
                    m_loss = self.rep_loss_fn(lbl2data_repr[idx], inputs_o.rep, inputs['lbl2data2ptr'][idx],
                                              inputs['idx'], inputs['plbl2data2ptr'][idx], inputs['pidx'])
                    loss += lw * m_loss

            elif 'data2ptr' in inputs:
                idx = torch.where(inputs['data2ptr'])[0]
                if len(idx) > 0:
                    inputs_o = encoder(data_input_ids=inputs['input_ids'], data_attention_mask=inputs['attention_mask'], 
                                       data_type="meta")
                    m_loss = self.rep_loss_fn(data_repr[idx], inputs_o.rep, inputs['data2ptr'][idx], inputs['idx'], 
                                              inputs['pdata2ptr'][idx], inputs['pidx'])
                    loss += lw * m_loss       

            else: raise ValueError('Invalid metadata input arguments.')
        return loss

    def compute_fusion_loss(self, data_repr, meta_repr:Dict, prefix:str, **kwargs):
        meta_inputs = Parameters.from_meta_pred_prefix(prefix, **kwargs)
        
        loss = 0.0
        if meta_repr is not None:
            for key,input_repr in meta_repr.items():
                inputs = meta_inputs[key]
                if 'lbl2data2ptr' in inputs:
                    idx = torch.where(inputs['lbl2data2ptr'])[0]
                    if len(idx) > 0:
                        m_loss = self.rep_loss_fn(data_repr[idx], input_repr, inputs['lbl2data2ptr'][idx],
                                                  inputs['idx'], inputs['plbl2data2ptr'][idx], inputs['pidx'])
                        loss += self.fusion_loss_weight * m_loss
    
                elif 'data2ptr' in inputs:
                    idx = torch.where(inputs['data2ptr'])[0]
                    if len(idx) > 0:
                        m_loss = self.rep_loss_fn(data_repr[idx], input_repr, inputs['data2ptr'][idx], inputs['idx'], 
                                                  inputs['pdata2ptr'][idx], inputs['pidx'])
                        loss += self.fusion_loss_weight * m_loss       
    
                else: raise ValueError('Invalid metadata input arguments.')
        return loss


    def get_meta_representation(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        **kwargs
    ):
        if self.use_encoder_parallel: 
            encoder = XCDataParallel(module=self.encoder)
        else: encoder = self.encoder
            
        data_o = encoder(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                         data_unnormalized=True, data_type="meta")
        return XCModelOutput(
            data_repr=data_o.rep,
            data_fused_repr=data_o.fused_rep,
        )

    
        
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):  
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.use_encoder_parallel: 
            encoder = XCDataParallel(module=self.encoder)
        else: encoder = self.encoder
        
        data_meta_kwargs = Parameters.from_feat_meta_aug_prefix('data', self.data_aug_meta_prefix, **kwargs)
        data_o = encoder(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                         data_aug_meta_prefix=self.data_aug_meta_prefix, **data_meta_kwargs)
        
        
        loss = None; lbl2data_o = EncoderOutput()
        if lbl2data_input_ids is not None:
            lbl2data_meta_kwargs = Parameters.from_feat_meta_aug_prefix('lbl2data', self.lbl2data_aug_meta_prefix, **kwargs)
            lbl2data_o = encoder(data_input_ids=lbl2data_input_ids, data_attention_mask=lbl2data_attention_mask, 
                                 data_aug_meta_prefix=self.lbl2data_aug_meta_prefix, **lbl2data_meta_kwargs)
            
            loss = self.compute_loss(data_o.fused_rep, lbl2data_o.rep,lbl2data_data2ptr,lbl2data_idx,
                                     plbl2data_data2ptr,plbl2data_idx)

            if self.use_query_loss:
                loss += self.compute_loss(data_o.rep, lbl2data_o.rep,lbl2data_data2ptr,lbl2data_idx,
                                          plbl2data_data2ptr,plbl2data_idx)

            if self.use_calib_loss:
                loss += self.calibration_loss(data_o.fused_rep, data_o.rep, lbl2data_o.rep,lbl2data_data2ptr,lbl2data_idx,
                                              plbl2data_data2ptr,plbl2data_idx)
            
            loss += self.compute_meta_loss(data_o.fused_rep, lbl2data_o.rep, **kwargs)
            
            if self.use_fusion_loss:
                loss += self.compute_fusion_loss(data_o.fused_rep, data_o.meta_repr, self.data_aug_meta_prefix, **kwargs)
                loss += self.compute_fusion_loss(lbl2data_o.rep, lbl2data_o.meta_repr, self.lbl2data_aug_meta_prefix, **kwargs)
            
            
        if not return_dict:
            o = (data_o.logits,data_o.rep,data_o.fused_rep,lbl2data_o.logits,lbl2data_o.rep,lbl2data_o.fused_rep)
            return ((loss,) + o) if loss is not None else o
        
        
        return XCModelOutput(
            loss=loss,
            
            data_repr=data_o.rep,
            data_fused_repr=data_o.fused_rep,
            
            lbl2data_repr=lbl2data_o.rep,
            lbl2data_fused_repr=lbl2data_o.fused_rep,
        )
        

# %% ../../nbs/20_models.oak.ipynb 36
class OAK001(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    @delegates(OAK000.__init__)
    def __init__(
        self, 
        config,
        num_metadata:int,
        resize_length:Optional[int]=None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.encoder = Encoder(config, num_metadata=num_metadata, resize_length=resize_length)
        self.post_init(); self.remap_post_init(); self.init_retrieval_head(); self.init_cross_head()

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert
        

# %% ../../nbs/20_models.oak.ipynb 47
class Encoder002(Encoder):

    def __init__(
        self, 
        config,
        cross_tau:Optional[float]=0.1, 
        cross_dropout:Optional[float]=0.1, 
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.cross_head = NormCrossAttention(config, tau=cross_tau, dropout=cross_dropout)
        self.post_init()
    

# %% ../../nbs/20_models.oak.ipynb 48
class OAK002(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    @delegates(OAK000.__init__)
    def __init__(
        self, 
        config,
        num_metadata:int,
        resize_length:Optional[int]=None,
        
        cross_tau:Optional[float]=0.1,
        cross_dropout:Optional[float]=0.1,
        
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.encoder = Encoder002(config, cross_tau=cross_tau, cross_dropout=cross_dropout,
                                  num_metadata=num_metadata, resize_length=resize_length)
        self.post_init(); self.remap_post_init(); self.init_retrieval_head(); self.init_cross_head()

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert
        

# %% ../../nbs/20_models.oak.ipynb 57
class Encoder003(Encoder):

    def __init__(
        self, 
        config,
        num_metadata:int,
        **kwargs
    ):
        super().__init__(config, num_metadata=num_metadata, **kwargs)
        self.pretrained_meta_embeddings = nn.Embedding(num_metadata, config.dim)
        self.post_init()

    def freeze_pretrained_meta_embeddings(self):
        self.pretrained_meta_embeddings.requires_grad_(False)

    def unfreeze_pretrained_meta_embeddings(self):
        self.pretrained_meta_embeddings.requires_grad_(True)

    def set_pretrained_meta_embeddings(self, embed:torch.Tensor):
        self.pretrained_meta_embeddings.weight.data = embed

    def init_meta_embeddings(self):
        self.meta_embeddings.weight.data = torch.zeros_like(self.meta_embeddings.weight.data)

    def fuse_meta_into_embeddings(self, data_repr:torch.Tensor, data_mask:torch.Tensor, meta_kwargs:Dict):
        meta_repr = {}
        
        data_fused_repr, data_mask = data_repr.clone().view(-1, 1, self.config.dim), data_mask.view(-1, 1)
        for m_key, m_args in meta_kwargs.items():
            idx = torch.where(m_args['data2ptr'] > 0)[0]
            meta_repr[m_key] = torch.empty(0, self.config.dim).to(data_repr)
            
            if len(idx):
                m_idx,m_repr_mask = self.resize(m_args['idx'], m_args['data2ptr'][idx])
                m_repr = F.normalize(self.meta_embeddings(m_idx) + self.pretrained_meta_embeddings(m_idx), dim=1)
                
                m_repr, m_repr_mask = m_repr.view(len(idx), -1, self.config.dim), m_repr_mask.bool().view(len(idx), -1)
                meta_repr[m_key] = m_repr[m_repr_mask]
                
                fused_repr = self.cross_head(data_fused_repr[idx], data_mask[idx], m_repr, m_repr_mask)[0]
                data_fused_repr[idx] += fused_repr
                
        return data_fused_repr.squeeze(), meta_repr
    

# %% ../../nbs/20_models.oak.ipynb 58
class OAK003(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    @delegates(OAK000.__init__)
    def __init__(
        self, 
        config,
        num_metadata:int,
        resize_length:Optional[int]=None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.encoder = Encoder003(config, num_metadata=num_metadata, resize_length=resize_length)
        self.post_init(); self.remap_post_init(); self.init_retrieval_head(); self.init_cross_head()

    def init_meta_embeddings(self):
        self.encoder.init_meta_embeddings()

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert
        
