# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_models.MMM00X.ipynb.

# %% auto 0
__all__ = ['XCModelOutput', 'Pooling', 'BT0001', 'BT0002', 'BT0003', 'BT0004', 'RT0005', 'BT0006']

# %% ../../nbs/05_models.MMM00X.ipynb 2
import torch, re, inspect
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Mapping, Any
from transformers import BertLMHeadModel, BatchEncoding, BertPreTrainedModel, BertModel, RobertaForCausalLM
from transformers.utils.generic import ModelOutput

from fastcore.meta import *

from ..losses import *

# %% ../../nbs/05_models.MMM00X.ipynb 9
@dataclass
class XCModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    dr_loss: Optional[torch.FloatTensor] = None
    data_repr: Optional[torch.FloatTensor] = None
    lbl2data_repr: Optional[torch.FloatTensor] = None
    data_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    data_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    data_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    lbl2data_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    lbl2data_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    lbl2data_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    

# %% ../../nbs/05_models.MMM00X.ipynb 11
class Pooling:

    @staticmethod
    def mean_pooling(data_embeds:torch.FloatTensor, data_attention_mask:torch.LongTensor):
        data_attention_mask = data_attention_mask.unsqueeze(2).expand(data_embeds.size()).float()
        return torch.sum(data_embeds * data_attention_mask, 1) / torch.clamp(data_attention_mask.sum(1), min=1e-9)


# %% ../../nbs/05_models.MMM00X.ipynb 15
class BT0001(BertLMHeadModel):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        data_token_type_ids:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_idx:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_token_type_ids:Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        data_o = self.bert(
            data_input_ids,
            data_attention_mask,
            data_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        data_logits = self.cls(data_o[0])
        data_repr = data_o[0].mean(dim=1)
        
        if lbl2data_input_ids is not None and lbl2data_data2ptr is not None:
            lbl2data_o = self.bert(
                lbl2data_input_ids,
                lbl2data_attention_mask,
                lbl2data_token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            lbl2data_repr = lbl2data_o[0].mean(dim=1)
            return data_logits, lbl2data_input_ids, lbl2data_data2ptr, lbl2data_idx, data_repr, lbl2data_repr, kwargs

        return data_logits, lbl2data_input_ids, lbl2data_data2ptr, lbl2data_idx, kwargs
        

# %% ../../nbs/05_models.MMM00X.ipynb 20
class BT0002(BertLMHeadModel):
    use_generation,use_representation = True,False 

    def __init__(self,
                 config,
                 tn_targ:Optional[int]=None, 
                 ig_tok:Optional[int]=0,
                 *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.loss_fn = MultiCrossEntropy(tn_targ=tn_targ, ig_tok=ig_tok, reduce='mean')

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        data_token_type_ids:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_token_type_ids:Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        data_o = self.bert(
            data_input_ids,
            data_attention_mask,
            data_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        data_logits = self.cls(data_o[0])
        
        loss = None
        if lbl2data_input_ids is not None:
            loss = self.loss_fn(data_logits, lbl2data_input_ids, lbl2data_data2ptr, **kwargs)

        if not return_dict:
            o = (data_logits,) + data_o[2:]
            return ((loss,) + o) if loss is not None else o

        return XCModelOutput(
            loss=loss,
            logits=data_logits,
            data_hidden_states=data_o.hidden_states,
            data_attentions=data_o.attentions,
            data_cross_attentions=data_o.cross_attentions,
        )
        

# %% ../../nbs/05_models.MMM00X.ipynb 28
class BT0003(BertPreTrainedModel):
    use_generation,use_representation = False,True
    
    def __init__(self,
                 config,
                 bsz:Optional[int]=None,
                 tn_targ:Optional[int]=None,
                 margin:Optional[float]=0.8,
                 ig_tok:Optional[int]=0,
                 *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = BertModel(config)
        self.loss_fn = MultiTriplet(bsz=bsz, tn_targ=tn_targ, margin=margin, ig_tok=ig_tok, reduce='mean')
        self.post_init()

    @delegates(BertModel.__call__)
    def get_repr(self, 
                 input_ids:Optional[torch.Tensor]=None, 
                 attention_mask:Optional[torch.Tensor]=None,
                 token_type_ids:Optional[torch.Tensor]=None,
                 **kwargs):
        o = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            **kwargs
        )
        return o, Pooling.mean_pooling(o[0], attention_mask)

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        data_token_type_ids:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_token_type_ids:Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        data_o, data_repr = self.get_repr(data_input_ids, 
                                          data_attention_mask, 
                                          data_token_type_ids, 
                                          output_attentions=output_attentions, 
                                          output_hidden_states=output_hidden_states,
                                          return_dict=return_dict)
        loss, lbl2data_repr = None, None
        if lbl2data_input_ids is not None:
            lbl2data_o, lbl2data_repr = self.get_repr(lbl2data_input_ids, 
                                                      lbl2data_attention_mask, 
                                                      lbl2data_token_type_ids, 
                                                      output_attentions=output_attentions, 
                                                      output_hidden_states=output_hidden_states,
                                                      return_dict=return_dict)
            loss = self.loss_fn(data_repr, lbl2data_repr, lbl2data_data2ptr, **kwargs)

        if not return_dict:
            o = (data_repr, lbl2data_repr)
            return ((loss,) + o) if loss is not None else o

        return XCModelOutput(
            loss=loss,
            data_repr=data_repr,
            lbl2data_repr=lbl2data_repr,
        )
        

# %% ../../nbs/05_models.MMM00X.ipynb 37
class BT0004(BertLMHeadModel):
    use_generation,use_representation = True,True 

    def __init__(self,
                 config,
                 bsz:Optional[int]=None,
                 tn_targ:Optional[int]=None, 
                 ig_tok:Optional[int]=0,
                 lw:Optional[int]=0.5,
                 *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.lw, self.dr_loss_fn = lw, SoupCon(bsz=bsz, reduce='mean')
        self.lm_loss_fn = MultiCrossEntropy(tn_targ=tn_targ, ig_tok=ig_tok, reduce='mean')
        
    @delegates(BertModel.__call__)
    def get_repr(self, 
                 input_ids:Optional[torch.Tensor]=None, 
                 attention_mask:Optional[torch.Tensor]=None,
                 token_type_ids:Optional[torch.Tensor]=None,
                 **kwargs):
        o = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            **kwargs
        )
        return o, F.normalize(o[0].mean(dim=1), dim=1)

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        data_token_type_ids:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_token_type_ids:Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        data_o, data_repr = self.get_repr(
            data_input_ids,
            data_attention_mask,
            data_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        data_logits = self.cls(data_o[0])
        
        loss, lm_loss, dr_loss, lbl2data_repr = None, None, None, None
        if lbl2data_input_ids is not None:
            lbl2data_o, lbl2data_repr = self.get_repr(
                lbl2data_input_ids,
                lbl2data_attention_mask,
                lbl2data_token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            lm_loss = self.lm_loss_fn(data_logits, lbl2data_input_ids, lbl2data_data2ptr)
            dr_loss = self.dr_loss_fn(data_repr, lbl2data_repr, lbl2data_data2ptr, **kwargs)
            loss = lm_loss + self.lw*dr_loss
            
        if not return_dict:
            o = (data_logits,data_repr,lbl2data_repr) + data_o[2:]
            return ((loss,lm_loss,dr_loss) + o) if loss is not None else o

        return XCModelOutput(
            loss=loss,
            lm_loss=lm_loss,
            dr_loss=dr_loss,
            logits=data_logits,
            data_repr=data_repr,
            lbl2data_repr=lbl2data_repr,
            data_hidden_states=data_o.hidden_states,
            data_attentions=data_o.attentions,
            data_cross_attentions=data_o.cross_attentions,
        )
        

# %% ../../nbs/05_models.MMM00X.ipynb 43
class RT0005(RobertaForCausalLM):
    use_generation,use_representation = True,False 

    def __init__(self,
                 config,
                 tn_targ:Optional[int]=None, 
                 ig_tok:Optional[int]=0,
                 *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.loss_fn = MultiCrossEntropy(tn_targ=tn_targ, ig_tok=ig_tok, reduce='mean')

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        data_o = self.roberta(
            data_input_ids,
            data_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        data_logits = self.lm_head(data_o[0])
        
        loss = None
        if lbl2data_input_ids is not None:
            loss = self.loss_fn(data_logits, lbl2data_input_ids, lbl2data_data2ptr, **kwargs)

        if not return_dict:
            o = (data_logits,) + data_o[2:]
            return ((loss,) + o) if loss is not None else o

        return XCModelOutput(
            loss=loss,
            logits=data_logits,
            data_hidden_states=data_o.hidden_states,
            data_attentions=data_o.attentions,
            data_cross_attentions=data_o.cross_attentions,
        )
        

# %% ../../nbs/05_models.MMM00X.ipynb 51
class BT0006(BertPreTrainedModel):
    use_generation,use_representation = False,True
    
    def __init__(self,
                 config,
                 bsz:Optional[int]=None,
                 *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = BertModel(config)
        self.loss_fn = SoupCon(bsz=bsz, reduce='mean')
        self.post_init()

    @delegates(BertModel.__call__)
    def get_repr(self, 
                 input_ids:Optional[torch.Tensor]=None, 
                 attention_mask:Optional[torch.Tensor]=None,
                 token_type_ids:Optional[torch.Tensor]=None,
                 **kwargs):
        o = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            **kwargs
        )
        return o, Pooling.mean_pooling(o[0], attention_mask)

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        data_token_type_ids:Optional[torch.Tensor]=None,
        lbl2data_data2ptr:Optional[torch.Tensor]=None,
        lbl2data_input_ids:Optional[torch.Tensor]=None,
        lbl2data_attention_mask:Optional[torch.Tensor]=None,
        lbl2data_token_type_ids:Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        data_o, data_repr = self.get_repr(data_input_ids, 
                                          data_attention_mask, 
                                          data_token_type_ids, 
                                          output_attentions=output_attentions, 
                                          output_hidden_states=output_hidden_states,
                                          return_dict=return_dict)
        loss, lbl2data_repr = None, None
        if lbl2data_input_ids is not None:
            lbl2data_o, lbl2data_repr = self.get_repr(lbl2data_input_ids, 
                                                      lbl2data_attention_mask, 
                                                      lbl2data_token_type_ids, 
                                                      output_attentions=output_attentions, 
                                                      output_hidden_states=output_hidden_states,
                                                      return_dict=return_dict)
            loss = self.loss_fn(data_repr, lbl2data_repr, lbl2data_data2ptr, **kwargs)

        if not return_dict:
            o = (data_repr, lbl2data_repr)
            return ((loss,) + o) if loss is not None else o

        return XCModelOutput(
            loss=loss,
            data_repr=data_repr,
            lbl2data_repr=lbl2data_repr,
        )
        
