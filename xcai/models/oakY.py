# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/33_models.oakY.ipynb.

# %% auto 0
__all__ = ['Encoder', 'OAK000', 'OAK001', 'Encoder002', 'OAK002', 'CrossAttention003', 'Encoder003', 'OAK003', 'Encoder004',
           'OAK004']

# %% ../../nbs/33_models.oakY.ipynb 2
import torch, re, inspect, pickle, os, torch.nn as nn, math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Mapping, Any, Union
from transformers import (
    PretrainedConfig,
    DistilBertForMaskedLM,
    DistilBertModel,
    DistilBertPreTrainedModel,
    DistilBertConfig,
)
from transformers.utils.generic import ModelOutput
from transformers.activations import get_activation

from fastcore.meta import *
from fastcore.utils import *

from ..losses import *
from ..core import store_attr
from ..learner import XCDataParallel
from .modeling_utils import *

# %% ../../nbs/33_models.oakY.ipynb 14
class Encoder(DistilBertPreTrainedModel):
    
    def __init__(
        self, 
        config:PretrainedConfig, 
    ):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        
        self.dr_head = RepresentationHead(config)
        self.dr_fused_head = RepresentationHead(config)
        self.meta_head = RepresentationHead(config)
        self.cross_head = CrossAttention(config)
        self.meta_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        self.post_init()

    def freeze_meta_embeddings(self):
        self.meta_embeddings.requires_grad_(False)

    def unfreeze_meta_embeddings(self):
        self.meta_embeddings.requires_grad_(True)

    def set_meta_embeddings(self, embed:torch.Tensor):
        with torch.no_grad():
            self.meta_embeddings.weight.copy_(embed)
        
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

    def dr_fused(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.dr_fused_head(embed)
        return F.normalize(Pooling.mean_pooling(embed, attention_mask), dim=1)

    def meta(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.meta_head(embed)
        return F.normalize(Pooling.mean_pooling(embed, attention_mask), dim=1)
    
    def meta_unnormalized(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.meta_head(embed)
        return Pooling.mean_pooling(embed, attention_mask)

    def fuse_meta_into_embeddings(self, embed:torch.Tensor, attention_mask:torch.Tensor, meta_kwargs:Dict):
        meta_repr = {}
        
        for m_key, m_args in meta_kwargs.items():
            idx = torch.where(m_args['data2ptr'] > 0)[0]
            meta_repr[m_key] = torch.empty(0, self.config.dim).to(embed)
            
            if len(idx):
                assert torch.all(m_args['data2ptr'][idx] == m_args['data2ptr'].max()), f'All datapoints should have same number of metadata.'
                
                if 'meta_repr' in m_args:
                    m_repr,m_repr_mask = m_args['meta_repr'],torch.any(m_args['attention_mask'], dim=1).long().view(-1,1)
                    m_repr_mask = m_repr_mask.bool()
                else:
                    m_input_ids, m_attention_mask = m_args['input_ids'], m_args['attention_mask']
                    m_embed = self.meta_embeddings(m_input_ids)
    
                    m_repr = self.meta_unnormalized(m_embed, m_attention_mask)
                    m_repr_mask = torch.any(m_attention_mask, dim=1)
                    
                m_repr, m_repr_mask = m_repr.view(len(idx), -1, self.config.dim), m_repr_mask.view(len(idx), -1)
                
                meta_repr[m_key] = m_repr[m_repr_mask]
                meta_repr[m_key] = F.normalize(meta_repr[m_key], dim=1)
                
                fused_embed, w = self.cross_head(embed[idx], attention_mask[idx], m_repr, m_repr_mask, output_attentions=True)
                embed[idx] += fused_embed
               
        return embed, meta_repr

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
                data_fused_embed, meta_repr = self.fuse_meta_into_embeddings(data_o[0], 
                                                                            data_attention_mask, 
                                                                            meta_kwargs)
                data_fused_repr = self.dr_fused(data_fused_embed, data_attention_mask)
                
        return EncoderOutput(
            rep=data_repr,
            fused_rep=data_fused_repr,
            meta_repr=meta_repr,
        )
        

# %% ../../nbs/33_models.oakY.ipynb 16
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
        assert self.encoder is not None, 'Encoder is not initialized.'
        self.encoder.dr_head.post_init()
        self.encoder.meta_head.post_init()
        self.encoder.dr_fused_head.post_init()

    def init_cross_head(self):
        assert self.encoder is not None, 'Encoder is not initialized.'
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
            
            # loss += self.compute_meta_loss(data_o.fused_rep, lbl2data_o.rep, **kwargs)
            
            # if self.use_fusion_loss:
            #    loss += self.compute_fusion_loss(data_o.fused_rep, data_o.meta_repr, self.data_aug_meta_prefix, **kwargs)
            #    loss += self.compute_fusion_loss(lbl2data_o.rep, lbl2data_o.meta_repr, self.lbl2data_aug_meta_prefix, **kwargs)
            
            
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
        

# %% ../../nbs/33_models.oakY.ipynb 18
class OAK001(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    @delegates(OAK000.__init__)
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = Encoder(config)
        self.post_init(); self.remap_post_init();

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert
        

# %% ../../nbs/33_models.oakY.ipynb 30
class Encoder002(Encoder):
    
    def __init__(
        self, 
        config:PretrainedConfig, 
    ):
        super().__init__(config)
        self.post_init()

    def fuse_meta_into_embeddings(self, embed:torch.Tensor, attention_mask:torch.Tensor, meta_kwargs:Dict):
        meta_repr = {}
        
        for m_key, m_args in meta_kwargs.items():
            idx = torch.where(m_args['data2ptr'] > 0)[0]
            meta_repr[m_key] = torch.empty(0, self.config.dim).to(embed)
            
            if len(idx):
                assert torch.all(m_args['data2ptr'][idx] == m_args['data2ptr'].max()), f'All datapoints should have same number of metadata.'
                
                if 'meta_repr' in m_args:
                    m_repr,m_repr_mask = m_args['meta_repr'],torch.any(m_args['attention_mask'], dim=1).long().view(-1,1)
                    m_repr_mask = m_repr_mask.bool()
                else:
                    m_input_ids, m_attention_mask = m_args['input_ids'], m_args['attention_mask']
                    m_embed = self.meta_embeddings(m_input_ids)

                m_repr, m_repr_mask = m_embed.view(len(idx), -1, self.config.dim), m_attention_mask.view(len(idx), -1)
                meta_repr[m_key] = self.meta(m_embed, m_attention_mask)
                
                fused_embed, w = self.cross_head(embed[idx], attention_mask[idx], m_repr, m_repr_mask, output_attentions=True)
                embed[idx] += fused_embed
               
        return embed, meta_repr
        

# %% ../../nbs/33_models.oakY.ipynb 31
class OAK002(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    @delegates(OAK000.__init__)
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = Encoder002(config)
        self.post_init(); self.remap_post_init();

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert
        

# %% ../../nbs/33_models.oakY.ipynb 44
class CrossAttention003(nn.Module):
    
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
        torch.nn.init.eye_(self.q.weight)
        torch.nn.init.eye_(self.k.weight)
        torch.nn.init.eye_(self.v.weight)
        torch.nn.init.eye_(self.o.weight)

    def forward(
        self, 
        q: torch.Tensor,
        k: torch.Tensor, 
        output_attentions:Optional[bool] = False,
    ):  
        bsz, dim, v = q.size(0), q.size(1), k
        h_dim = self.dim//self.n_h

        def shape(x: torch.Tensor): return x.view(bsz, -1, self.n_h, h_dim).transpose(1, 2)
        def unshape(x: torch.Tensor): return x.transpose(1, 2).contiguous().view(bsz, -1, self.n_h * h_dim)

        q = shape(self.q(q))  # (bs, n_h, q_len, h_dim)
        k = shape(self.k(k))  # (bs, n_h, k_len, h_dim)
        v = shape(self.v(v))  # (bs, n_h, k_len, h_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        o = unshape(y)

        return o.squeeze(1)
        

# %% ../../nbs/33_models.oakY.ipynb 45
class Encoder003(DistilBertPreTrainedModel):
    
    def __init__(
        self, 
        config:PretrainedConfig, 
    ):
        super().__init__(config)
        self.dr_distilbert = DistilBertModel(config)
        self.meta_distilbert = DistilBertModel(config)
        
        self.dr_head = RepresentationHead(config)
        self.cross_head = CrossAttention003(config)
        
        self.post_init()
    
    def dr_encode(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, **kwargs):
        o = self.dr_distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return Pooling.mean_pooling(o[0], attention_mask)

    def meta_encode(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, **kwargs):
        o = self.meta_distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return Pooling.mean_pooling(o[0], attention_mask)

    def dr(self, embed:torch.Tensor):
        return F.normalize(self.dr_head(embed), dim=1)

    def init_meta_encoder(self):
        sd_meta, sd_dr = self.meta_distilbert.state_dict(), self.dr_distilbert.state_dict()
        for k in sd_dr:
            assert sd_meta[k].shape == sd_dr[k].shape
            with torch.no_grad():
                sd_meta[k].copy_(sd_dr[k])

    def fuse_meta_into_embeddings(self, embed:torch.Tensor, meta_kwargs:Dict):
        meta_repr, bsz = {}, embed.size(0)
        
        for m_key, m_args in meta_kwargs.items():
            n_meta = m_args['data2ptr'].max()
            assert torch.all(m_args['data2ptr'] == n_meta), f'All datapoints should have same number of metadata.'
            
            m_input_ids, m_attention_mask = m_args['input_ids'], m_args['attention_mask']
            m_embed = self.meta_encode(input_ids=m_input_ids, attention_mask=m_attention_mask)
            meta_repr[m_key] = m_embed
                    
            m_embed = m_embed.view(bsz, -1, self.config.dim)  
            fused_embed = self.cross_head(embed, m_embed)
            embed = embed + fused_embed
               
        return embed, meta_repr

    def forward(
        self, 
        data_input_ids: torch.Tensor, 
        data_attention_mask: torch.Tensor,
        data_aug_meta_prefix: Optional[str]=None,
        data_type:Optional[str]=None,
        data_unnormalized:Optional[bool]=False,
        **kwargs
    ):  
        data_repr = self.dr_encode(data_input_ids, data_attention_mask)
        
        data_fused_repr = meta_repr = None
        if data_aug_meta_prefix is not None:
            meta_kwargs = Parameters.from_meta_aug_prefix(data_aug_meta_prefix, **kwargs)
            if len(meta_kwargs):
                data_fused_repr, meta_repr = self.fuse_meta_into_embeddings(data_repr, meta_kwargs)
                data_fused_repr = self.dr(data_fused_repr)
                
        return EncoderOutput(
            rep=data_repr,
            fused_rep=data_fused_repr,
            meta_repr=meta_repr,
        )
        

# %% ../../nbs/33_models.oakY.ipynb 46
class OAK003(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.dr_distilbert"]
    _keys_to_ignore_on_load_missing = ["encoder.meta_distilbert"]

    @delegates(OAK000.__init__)
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = Encoder003(config)
        self.post_init(); self.remap_post_init();

    def init_retrieval_head(self):
        self.encoder.dr_head.post_init()

    def init_meta_encoder(self):
        self.encoder.init_meta_encoder()

    def remap_post_init(self):
        self.distilbert = self.encoder.dr_distilbert
        

# %% ../../nbs/33_models.oakY.ipynb 61
class Encoder004(DistilBertPreTrainedModel):
    
    def __init__(
        self, 
        config:PretrainedConfig, 
    ):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dr_head = RepresentationHead(config)
        self.cross_head = CrossAttention003(config)
        self.post_init()
    
    def encode(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, **kwargs):
        o = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return Pooling.mean_pooling(o[0], attention_mask)

    def dr(self, embed:torch.Tensor):
        return F.normalize(self.dr_head(embed), dim=1)

    def fuse_meta_into_embeddings(self, embed:torch.Tensor, meta_kwargs:Dict):
        meta_repr, bsz = {}, embed.size(0)
        
        for m_key, m_args in meta_kwargs.items():
            n_meta = m_args['data2ptr'].max()
            assert torch.all(m_args['data2ptr'] == n_meta), f'All datapoints should have same number of metadata.'
            
            m_embed = m_args['meta_repr']
            meta_repr[m_key] = m_embed
                    
            m_embed = m_embed.view(bsz, -1, self.config.dim)  
            fused_embed = self.cross_head(embed, m_embed)
            embed = embed + fused_embed
               
        return embed, meta_repr

    def forward(
        self, 
        data_input_ids: torch.Tensor, 
        data_attention_mask: torch.Tensor,
        data_aug_meta_prefix: Optional[str]=None,
        data_type:Optional[str]=None,
        data_unnormalized:Optional[bool]=False,
        **kwargs
    ):  
        data_repr = self.encode(data_input_ids, data_attention_mask)
        
        data_fused_repr = meta_repr = None
        if data_aug_meta_prefix is not None:
            meta_kwargs = Parameters.from_meta_aug_prefix(data_aug_meta_prefix, **kwargs)
            if len(meta_kwargs):
                data_fused_repr, meta_repr = self.fuse_meta_into_embeddings(data_repr, meta_kwargs)
                data_fused_repr = self.dr(data_fused_repr)
                
        return EncoderOutput(
            rep=data_repr,
            fused_rep=data_fused_repr,
            meta_repr=meta_repr,
        )
        

# %% ../../nbs/33_models.oakY.ipynb 62
class OAK004(OAK000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    @delegates(OAK000.__init__)
    def __init__(self, config, num_metadata:int, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = Encoder004(config)
        self.meta_embeddings = nn.Embedding(num_metadata, config.dim, sparse=True)
        self.post_init(); self.remap_post_init();

    def init_retrieval_head(self):
        self.encoder.dr_head.post_init()

    def init_meta_embeddings(self):
        torch.nn.init.zeros_(self.meta_embeddings.weight)

    def freeze_meta_embeddings(self):
        self.meta_embeddings.requires_grad_(False)

    def unfreeze_meta_embeddings(self):
        self.meta_embeddings.requires_grad_(True)

    def set_meta_embeddings(self, embed:torch.Tensor):
        with torch.no_grad():
            self.meta_embeddings.weight.copy_(embed)

    def _get_encoder_meta_kwargs(self, feat:str, prefix:str, **kwargs):
        meta_kwargs = Parameters.from_feat_meta_aug_prefix(feat, prefix, **kwargs)
        if f'{prefix}_idx' in meta_kwargs:
            m_idx = meta_kwargs[f'{prefix}_idx']
            if len(m_idx): meta_kwargs[f'{prefix}_meta_repr'] = self.meta_embeddings(m_idx)
        return meta_kwargs

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert

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
        
        data_meta_kwargs = self._get_encoder_meta_kwargs('data', self.data_aug_meta_prefix, **kwargs)
        data_o = encoder(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                         data_aug_meta_prefix=self.data_aug_meta_prefix, **data_meta_kwargs)
        
        
        loss = None; lbl2data_o = EncoderOutput()
        if lbl2data_input_ids is not None:
            lbl2data_meta_kwargs = self._get_encoder_meta_kwargs('lbl2data', self.lbl2data_aug_meta_prefix, **kwargs)
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
        
        
