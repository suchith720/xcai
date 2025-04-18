# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/37_models.cachew.ipynb.

# %% auto 0
__all__ = ['Parameters', 'MemoryConfig', 'CachewConfig', 'Memory', 'CrossCombinerBlock', 'EncoderOutput', 'Encoder',
           'CAWModelOutput', 'CAW000', 'CAW001', 'CAW002']

# %% ../../nbs/37_models.cachew.ipynb 3
import torch, torch.nn as nn, torch.nn.functional as F, re
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from transformers.utils.generic import ModelOutput
from transformers import PretrainedConfig, DistilBertConfig, DistilBertPreTrainedModel, DistilBertModel
from transformers.models.distilbert.modeling_distilbert import create_sinusoidal_embeddings, TransformerBlock

from ..losses import *
from ..learner import XCDataParallel
from .modeling_utils import *

# %% ../../nbs/37_models.cachew.ipynb 16
class Parameters:
    
    @staticmethod
    def from_data_aug_meta_prefix_for_encoder(prefix:str, **kwargs):
        inputs = {}
        args = [arg for arg in kwargs if prefix is not None and re.match(f'^{prefix}.*_(input_ids|attention_mask|data2ptr|idx)$', arg)]
        for arg in args:
            meta,param = arg.split('_', maxsplit=1)
            inputs.setdefault(meta, {})[param] = kwargs[arg]
        return inputs
    
    @staticmethod
    def from_aug_meta_prefix_for_feature(feat:str, prefix:str, **kwargs):
        keys = ['attention_mask', 'input_ids', 'idx']        
        inputs = {f'{prefix}_{k}': kwargs[f'{prefix}_{k}'] for k in keys if f'{prefix}_{k}' in kwargs}
        if prefix is not None and f'{prefix}_{feat}2ptr' in kwargs:
            inputs.update({f'{prefix}_data2ptr': kwargs[f'{prefix}_{feat}2ptr']})
        return inputs

    @staticmethod
    def from_aug_meta_prefix_for_loss(feat:str, prefix:str, **kwargs):
        keys = [f'{prefix}_idx', f'p{prefix}_idx']
        args = {k: kwargs[k] for k in keys if k in kwargs}
        if prefix is not None and f'{prefix}_{feat}2ptr' in kwargs:
            args.update({f'{prefix}_data2ptr': kwargs[f'{prefix}_{feat}2ptr']})
        if prefix is not None and f'p{prefix}_{feat}2ptr' in kwargs:
            args.update({f'p{prefix}_data2ptr': kwargs[f'p{prefix}_{feat}2ptr']})

        inputs = {}
        for arg in args:
            meta,param = arg.split('_', maxsplit=1)
            inputs.setdefault(meta, {})[param] = args[arg]
        return inputs
        

# %% ../../nbs/37_models.cachew.ipynb 21
class MemoryConfig(DistilBertConfig):

    def __init__(
        self,
        top_k_metadata:Optional[int] = 5,
        num_metadata:Optional[int] = 100_000,
        **kwargs,
    ):
        self.top_k_metadata = top_k_metadata
        self.num_metadata = num_metadata
        super().__init__(**kwargs)
    

# %% ../../nbs/37_models.cachew.ipynb 22
class CachewConfig(MemoryConfig):

    def __init__(
        self,
        data_aug_meta_prefix:Optional[str] = None, 
        lbl2data_aug_meta_prefix:Optional[str] = None,

        data_enrich:Optional[bool] = True,
        lbl2data_enrich:Optional[bool] = True,
        
        num_batch_labels:Optional[int] = None,
        batch_size:Optional[int] = None,
        margin:Optional[float] = 0.3,
        num_negatives:Optional[int] = 10,
        tau:Optional[float] = 0.1,
        apply_softmax:Optional[bool] = True,

        calib_margin:Optional[float] = 0.05,
        calib_num_negatives:Optional[int] = 10,
        calib_tau:Optional[float] = 0.1,
        calib_apply_softmax:Optional[bool] = False,
        calib_loss_weight:Optional[float] = 0.1,
        use_calib_loss:Optional[float] = False,
        
        use_query_loss:Optional[float] = True,

        meta_loss_weight:Optional[float] = 0.1,
        use_meta_loss:Optional[bool] = False,
        
        use_encoder_parallel:Optional[bool] = True,
        
        use_self_linker:Optional[bool] = True,
        
        **kwargs,
    ):
        self.data_aug_meta_prefix = data_aug_meta_prefix
        self.lbl2data_aug_meta_prefix = lbl2data_aug_meta_prefix

        self.data_enrich = data_enrich
        self.lbl2data_enrich = lbl2data_enrich

        self.num_batch_labels = num_batch_labels
        self.batch_size = batch_size
        self.margin = margin
        self.num_negatives = num_negatives
        self.tau = tau
        self.apply_softmax = apply_softmax

        self.calib_margin = calib_margin
        self.calib_num_negatives = calib_num_negatives
        self.calib_tau = calib_tau
        self.calib_apply_softmax = calib_apply_softmax
        self.calib_loss_weight = calib_loss_weight
        self.use_calib_loss = use_calib_loss

        self.use_query_loss = use_query_loss

        self.meta_loss_weight = meta_loss_weight
        self.use_meta_loss = use_meta_loss

        self.use_encoder_parallel = use_encoder_parallel
        
        self.use_self_linker = use_self_linker
        
        super().__init__(**kwargs)
        

# %% ../../nbs/37_models.cachew.ipynb 26
class Memory(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.top_k_metadata = config.top_k_metadata
        
        self.memory_embeddings = nn.Embedding(config.num_metadata, config.dim)
        
        self.position_embeddings = nn.Embedding(config.num_metadata, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.num_metadata, dim=config.dim, out=self.position_embeddings.weight
            )
        
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def set_memory_embeddings(self, embed:torch.Tensor):
        with torch.no_grad():
            self.memory_embeddings.weight.copy_(embed)

    def align_embeddings(self, embeddings:torch.Tensor, group_lengths:torch.Tensor):
        n, dim = embeddings.shape
        num_groups, max_len = len(group_lengths), group_lengths.max()
        group_ids = torch.repeat_interleave(torch.arange(num_groups, device=embeddings.device), group_lengths)

        row_indices = torch.arange(n, device=embeddings.device)

        group_start = torch.cat([torch.zeros(1, dtype=group_lengths.dtype, device=group_lengths.device), group_lengths.cumsum(0)[:-1]], dim=0)

        within_idx = row_indices - group_start[group_ids]

        output, mask = torch.zeros((num_groups, max_len, dim), device=embeddings.device), torch.zeros((num_groups, max_len), device=embeddings.device)
        output[group_ids, within_idx] = embeddings
        mask[group_ids, within_idx] = 1.0

        return output, mask
        
    def forward(self, input_embeds:Optional[torch.Tensor]=None, input_indices:Optional[torch.Tensor]=None, input_data2ptr:Optional[torch.Tensor]=None):
        pred_embeddings = pred_mask = scores = None
        if input_embeds is not None:
            assert input_embeds.dim() == 2, f'Input embeddings should be 2-dimensional, but got dim:{input_embeds.dim()}'
            
            meta_norm = F.normalize(self.memory_embeddings.weight, dim=-1)
            input_norm = F.normalize(input_embeds, dim=-1)
            
            scores = input_norm@meta_norm.T
            values, indices = torch.topk(scores, self.top_k_metadata, dim=-1)
            
            pred_embeddings = self.memory_embeddings(indices) + self.position_embeddings(indices)
            pred_embeddings = self.LayerNorm(pred_embeddings)
            pred_embeddings = self.dropout(pred_embeddings)
            pred_mask = torch.ones(pred_embeddings.shape[0], pred_embeddings.shape[1], device=pred_embeddings.device)
            
        input_embeddings = input_mask = None
        if input_indices is not None:
            input_embeddings = self.memory_embeddings(input_indices) + self.position_embeddings(input_indices)
            input_embeddings = self.LayerNorm(input_embeddings)
            input_embeddings = self.dropout(input_embeddings)
            input_embeddings, input_mask = self.align_embeddings(input_embeddings, input_data2ptr)

        if input_embeddings is None:
            embeddings, mask = pred_embeddings, pred_mask
        elif pred_embeddings is None: 
            embeddings, mask = input_embeddings, input_mask
        else:
            embeddings, mask = torch.cat([pred_embeddings, input_embeddings], dim=1), torch.cat([pred_mask, input_mask], dim=1)
            
        return embeddings, mask, scores
        

# %% ../../nbs/37_models.cachew.ipynb 35
class CrossCombinerBlock(TransformerBlock):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

    def post_init(self):
        for module in self.modules(): self._init_weights(module)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.eye_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        # Cross-Attention
        ca_output = self.attention(
            query=x,
            key=m,
            value=m,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            ca_output, ca_weights = ca_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            if type(ca_output) is not tuple:
                raise TypeError(f"ca_output must be a tuple but it is {type(ca_output)} type")

            ca_output = ca_output[0]
        ca_output = self.sa_layer_norm(ca_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(ca_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + ca_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (ca_weights,) + output
        return output
        

# %% ../../nbs/37_models.cachew.ipynb 37
@dataclass
class EncoderOutput(ModelOutput):
    repr: Optional[torch.FloatTensor] = None
    enriched_repr: Optional[torch.FloatTensor] = None
    meta_scores: Optional[torch.FloatTensor] = None
    

# %% ../../nbs/37_models.cachew.ipynb 38
class Encoder(DistilBertPreTrainedModel):
    
    config_class = MemoryConfig
    
    def __init__(
        self, 
        config:PretrainedConfig, 
    ):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.query_head = RepresentationHead(config)
        self.combiner_head = CrossCombinerBlock(config)
        self.enriched_query_head = RepresentationHead(config)

        self.memory = Memory(config)
        
        self.post_init()

    @torch.no_grad()
    def init_heads_to_identity(self):
        self.query_head.post_init()
        self.combiner_head.post_init()
        self.enriched_query_head.post_init()

    @torch.no_grad()
    def init_combiner_to_last_layer(self):
        lsd = self.distilbert.transformer.layer[-1].state_dict()
        lsd_keys = lsd.keys()
        csd = self.combiner_head.state_dict()
        csd_keys = csd.keys()
        
        assert len(lsd_keys) == len(csd_keys), f'mismatched keys: {len(lsd_keys)} != {len(csd_keys)}'
        
        for k in csd_keys:
            assert csd[k].shape == lsd[k].shape
            csd[k].copy_(lsd[k])
            
    @torch.no_grad()
    def set_memory_embeddings(self, embed:torch.Tensor):
        self.memory.set_memory_embeddings(embed)
        
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
        
    def encode_query(self, embed:torch.Tensor, attention_mask:torch.Tensor):
        embed = self.query_head(embed)
        return F.normalize(Pooling.mean_pooling(embed, attention_mask), dim=1)

    def encode_enriched_query(self, embed:torch.Tensor):
        return F.normalize(self.enriched_query_head(embed), dim=1)

    def enrich_query_representation(self, data_repr:torch.Tensor, meta_kwargs:Optional[Dict]=None):
        if self.config.use_self_linker:
            meta_repr, meta_mask, meta_scores = self.memory(data_repr) if meta_kwargs is None else self.memory(data_repr, meta_kwargs['idx'], meta_kwargs['data2ptr'])
        else:
            meta_repr, meta_mask, meta_scores = self.memory(input_indices=meta_kwargs['idx'], input_data2ptr=meta_kwargs['data2ptr'])
            
        if meta_repr is None: raise ValueError('Metadata representation is None.')
            
        meta_mask = meta_mask.view(len(meta_mask), 1, 1, -1).bool()
        fusion_repr = self.combiner_head(x=data_repr.view(len(data_repr), 1, -1), m=meta_repr, attn_mask=meta_mask)
        fusion_repr = fusion_repr[0].squeeze(dim=1)
        
        enriched_data_repr = self.encode_enriched_query(data_repr + fusion_repr)
        return enriched_data_repr, meta_scores

    def forward(
        self, 
        data_input_ids: torch.Tensor, 
        data_attention_mask: torch.Tensor,
        data_aug_meta_prefix: Optional[str]=None,
        data_enrich: Optional[bool]=True,
        **kwargs
    ):  
        data_o = self.encode(data_input_ids, data_attention_mask)
        data_repr = self.encode_query(data_o[0], data_attention_mask)
        
        enriched_data_repr = meta_scores = None
        meta_kwargs = Parameters.from_data_aug_meta_prefix_for_encoder(data_aug_meta_prefix, **kwargs)
        meta_kwargs = meta_kwargs.get(data_aug_meta_prefix, None)
        
        if data_enrich:
            enriched_data_repr, meta_scores = self.enrich_query_representation(data_repr, meta_kwargs)
            
        return EncoderOutput(
            repr=data_repr,
            enriched_repr=enriched_data_repr,
            meta_scores=meta_scores
        )
        

# %% ../../nbs/37_models.cachew.ipynb 45
@dataclass
class CAWModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    data_repr: Optional[torch.FloatTensor] = None
    data_enriched_repr: Optional[torch.FloatTensor] = None
    lbl2data_repr: Optional[torch.FloatTensor] = None
    lbl2data_enriched_repr: Optional[torch.FloatTensor] = None
    

# %% ../../nbs/37_models.cachew.ipynb 46
class CAW000(nn.Module):

    config_class = CachewConfig
    
    def __init__(
        self, 
        config: CachewConfig,
    ):
        super().__init__(config)
        self.config, self.encoder = config, None
        self.rep_loss_fn = MultiTriplet(margin=config.margin, n_negatives=config.num_negatives, tau=config.tau, 
                                        apply_softmax=config.apply_softmax, reduce='mean')
        self.cab_loss_fn = Calibration(margin=config.calib_margin, tau=config.calib_tau, n_negatives=config.calib_num_negatives, 
                                       apply_softmax=config.calib_apply_softmax, reduce='mean')
        
    def init_heads_to_identity(self):
        if self.encoder is None: raise ValueError('Encoder not initialized.')
        self.encoder.init_heads_to_identity()

    def init_combiner_to_last_layer(self):
        if self.encoder is None: raise ValueError('Encoder not initialized.')
        self.encoder.init_combiner_to_last_layer()

    def set_memory_embeddings(self, embed:torch.Tensor):
        if self.encoder is None: raise ValueError('Encoder not initialized.')
        self.encoder.set_memory_embeddings(embed)
        
    def compute_loss(self, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx):
        return self.rep_loss_fn(inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx)

    def calibration_loss(self, einp_repr, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx):
        return self.config.calib_loss_weight * self.cab_loss_fn(einp_repr, inp_repr, targ_repr, targ_ptr, targ_idx, ptarg_ptr, ptarg_idx)

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
        
        if self.config.use_encoder_parallel: 
            encoder = XCDataParallel(module=self.encoder)
        else: encoder = self.encoder
        
        data_meta_kwargs = Parameters.from_aug_meta_prefix_for_feature('data', self.config.data_aug_meta_prefix, **kwargs)
        data_o = encoder(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                         data_aug_meta_prefix=self.config.data_aug_meta_prefix, data_enrich=self.config.data_enrich, **data_meta_kwargs)
        
        loss = None; lbl2data_o = EncoderOutput()
        if lbl2data_input_ids is not None:
            lbl2data_meta_kwargs = Parameters.from_aug_meta_prefix_for_feature('lbl', self.config.lbl2data_aug_meta_prefix, **kwargs)
            lbl2data_o = encoder(data_input_ids=lbl2data_input_ids, data_attention_mask=lbl2data_attention_mask, 
                                 data_aug_meta_prefix=self.config.lbl2data_aug_meta_prefix, data_enrich=self.config.lbl2data_enrich, **lbl2data_meta_kwargs)
            
            loss = self.compute_loss(data_o.enriched_repr, lbl2data_o.repr,lbl2data_data2ptr,lbl2data_idx,
                                     plbl2data_data2ptr,plbl2data_idx)

            if self.config.use_query_loss:
                loss += self.compute_loss(data_o.repr, lbl2data_o.repr,lbl2data_data2ptr,lbl2data_idx,
                                          plbl2data_data2ptr,plbl2data_idx)

            if self.config.use_calib_loss:
                loss += self.calibration_loss(data_o.enriched_repr, data_o.repr, lbl2data_o.repr,lbl2data_data2ptr,lbl2data_idx,
                                              plbl2data_data2ptr,plbl2data_idx)
            
        if not return_dict:
            o = (data_o.repr,data_o.enriched_repr,lbl2data_o.repr,lbl2data_o.enriched_repr)
            return ((loss,) + o) if loss is not None else o
        
        return CAWModelOutput(
            loss=loss,
            data_repr=data_o.repr,
            data_enriched_repr=data_o.enriched_repr,
            lbl2data_repr=lbl2data_o.repr,
            lbl2data_enriched_repr=lbl2data_o.enriched_repr,
        )
        

# %% ../../nbs/37_models.cachew.ipynb 48
class CAW001(CAW000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        
        self.post_init()
        self.remap_post_init()

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert
        

# %% ../../nbs/37_models.cachew.ipynb 80
class CAW002(CAW000, DistilBertPreTrainedModel):
    use_generation,use_representation = False,True
    _tied_weights_keys = ["encoder.distilbert"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.meta_loss_fn = MultiTripletFromScores(margin=config.margin, n_negatives=config.num_negatives, tau=config.tau, 
                                                  apply_softmax=config.apply_softmax, reduce='mean')
        self.post_init()
        self.remap_post_init()

    def remap_post_init(self):
        self.distilbert = self.encoder.distilbert

    def freeze_query_encoder(self):
        module_names = ['distilbert', 'query_head']
        for name in module_names:
            module = getattr(self.encoder, name)
            for param in module.parameters(): param.requires_grad_(False)

    def unfreeze_query_encoder(self):
        module_names = ['distilbert', 'query_head']
        for name in module_names:
            module = getattr(self.encoder, name)
            for param in module.parameters(): param.requires_grad_(True)

    def compute_meta_loss(self, scores, feat, prefix, **kwargs):
        loss = 0.0
        meta_kwargs = Parameters.from_aug_meta_prefix_for_loss(feat, prefix, **kwargs)
        if len(meta_kwargs):
            args, pargs = meta_kwargs[prefix], meta_kwargs[f'p{prefix}']
            loss = self.config.meta_loss_weight * self.meta_loss_fn(scores[:, args['idx']], args['data2ptr'], args['idx'], 
                                                                    pargs['data2ptr'], pargs['idx'])
        return loss

    def get_label_representation(
        self,
        data_idx:Optional[torch.Tensor]=None,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        **kwargs
    ):
        if self.use_encoder_parallel: 
            encoder = XCDataParallel(module=self.encoder)
        else: encoder = self.encoder

        meta_prefix = None
        if self.config.lbl2data_aug_meta_prefix is not None:
            meta_prefix = self.config.lbl2data_aug_meta_prefix.split('2')[0]
            meta_prefix = f'{meta_prefix}2data'
        
        meta_kwargs = Parameters.from_aug_meta_prefix_for_feature('data', meta_prefix, **kwargs)
        data_o = encoder(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, data_aug_meta_prefix=meta_prefix, 
                         data_enrich=self.config.lbl2data_enrich, **meta_kwargs)
        
        return CAWModelOutput(
            data_repr=data_o.repr,
            data_enriched_repr=data_o.enriched_repr,
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
        
        if self.config.use_encoder_parallel: 
            encoder = XCDataParallel(module=self.encoder)
        else: encoder = self.encoder
        
        data_meta_kwargs = Parameters.from_aug_meta_prefix_for_feature('data', self.config.data_aug_meta_prefix, **kwargs)
        data_o = encoder(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                         data_aug_meta_prefix=self.config.data_aug_meta_prefix, data_enrich=self.config.data_enrich, **data_meta_kwargs)
        
        loss = None; lbl2data_o = EncoderOutput()
        if lbl2data_input_ids is not None:
            lbl2data_meta_kwargs = Parameters.from_aug_meta_prefix_for_feature('lbl', self.config.lbl2data_aug_meta_prefix, **kwargs)
            lbl2data_o = encoder(data_input_ids=lbl2data_input_ids, data_attention_mask=lbl2data_attention_mask, 
                                 data_aug_meta_prefix=self.config.lbl2data_aug_meta_prefix, data_enrich=self.config.lbl2data_enrich, **lbl2data_meta_kwargs)
            
            loss = self.compute_loss(data_o.enriched_repr, lbl2data_o.repr,lbl2data_data2ptr,lbl2data_idx,
                                     plbl2data_data2ptr,plbl2data_idx)
            
            if self.config.use_query_loss:
                loss += self.compute_loss(data_o.repr, lbl2data_o.repr,lbl2data_data2ptr,lbl2data_idx,
                                          plbl2data_data2ptr,plbl2data_idx)
                
            if self.config.use_calib_loss:
                loss += self.calibration_loss(data_o.enriched_repr, data_o.repr, lbl2data_o.repr,lbl2data_data2ptr,lbl2data_idx,
                                              plbl2data_data2ptr,plbl2data_idx)

            if self.config.use_meta_loss:
                loss += self.compute_meta_loss(data_o.meta_scores, 'data', self.config.data_aug_meta_prefix, **kwargs)
                loss += self.compute_meta_loss(lbl2data_o.meta_scores, 'lbl', self.config.lbl2data_aug_meta_prefix, **kwargs)
            
        if not return_dict:
            o = (data_o.repr,data_o.enriched_repr,lbl2data_o.repr,lbl2data_o.enriched_repr)
            return ((loss,) + o) if loss is not None else o
        
        return CAWModelOutput(
            loss=loss,
            data_repr=data_o.repr,
            data_enriched_repr=data_o.enriched_repr,
            lbl2data_repr=lbl2data_o.repr,
            lbl2data_enriched_repr=lbl2data_o.enriched_repr,
        )
        
