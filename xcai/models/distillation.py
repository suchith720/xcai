# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/17_models.distillation.ipynb.

# %% auto 0
__all__ = ['TCHOutput', 'TCH001', 'TCH002', 'DTL001', 'DTL002', 'DTL003', 'DTL004']

# %% ../../nbs/17_models.distillation.ipynb 2
import torch, numpy as np
from typing import Optional
import torch.nn as nn
from dataclasses import dataclass

from ..core import store_attr
from ..losses import Cosine, MultiTriplet
from .PPP0XX import XCModelOutput
from .oak import OAK001
from .radga import RADOutput
from transformers import DistilBertPreTrainedModel,DistilBertConfig
from transformers.utils.generic import ModelOutput

# %% ../../nbs/17_models.distillation.ipynb 27
@dataclass
class TCHOutput(ModelOutput):
    data_repr: Optional[torch.FloatTensor] = None
    lbl2data_repr: Optional[torch.FloatTensor] = None
    

# %% ../../nbs/17_models.distillation.ipynb 28
class TCH001(DistilBertPreTrainedModel):

    def __init__(self, config, n_data:int, n_lbl:int, **kwargs):
        super().__init__(config, **kwargs)
        store_attr('n_data,n_lbl')
        self.data_repr = nn.Embedding(self.n_data, config.dim)
        self.lbl_repr = nn.Embedding(self.n_lbl, config.dim)

    def init_embeddings(self, data_repr:torch.Tensor, lbl_repr:torch.Tensor):
        self.data_repr.weight.data = data_repr
        self.lbl_repr.weight.data = lbl_repr

    def freeze_embeddings(self):
        self.data_repr.requires_grad_(False)
        self.lbl_repr.requires_grad_(False)

    def freeze_data_embeddings(self):
        self.data_repr.requires_grad_(False)

    def unfreeze_embeddings(self):
        self.data_repr.requires_grad_(True)
        self.lbl_repr.requires_grad_(True)

    def forward(
        self,
        data_idx:torch.Tensor,
        lbl2data_idx:torch.Tensor,
    ):
        return TCHOutput(
            data_repr=self.data_repr(data_idx),
            lbl2data_repr= self.lbl_repr(lbl2data_idx),
        )
        

# %% ../../nbs/17_models.distillation.ipynb 43
class TCH002(DistilBertPreTrainedModel):

    def __init__(self, config, n_data:int, n_lbl:int, **kwargs):
        super().__init__(config, **kwargs)
        store_attr('n_data,n_lbl')
        self.data_repr = nn.Embedding(self.n_data, config.dim)
        self.lbl_repr = nn.Embedding(self.n_lbl, config.dim)
        
        self.lbl_embeddings = nn.Embedding(self.n_lbl, config.dim)

    def init_representations(self, data_repr:torch.Tensor, lbl_repr:torch.Tensor):
        self.data_repr.weight.data = data_repr
        self.lbl_repr.weight.data = lbl_repr

    def init_lbl_embeddings(self):
        self.lbl_embeddings.weight.data = torch.zeros_like(self.lbl_repr.weight.data, dtype=torch.float32)

    def freeze_representations(self):
        self.data_repr.requires_grad_(False)
        self.lbl_repr.requires_grad_(False)

    def unfreeze_representations(self):
        self.data_repr.requires_grad_(True)
        self.lbl_repr.requires_grad_(True)

    def forward(
        self,
        data_idx:torch.Tensor,
        lbl2data_idx:torch.Tensor,
    ):
        data_repr = self.data_repr(data_idx)
        lbl2data_repr = self.lbl_repr(lbl2data_idx) + self.lbl_embeddings(lbl2data_idx)
        return TCHOutput(
            data_repr=data_repr,
            lbl2data_repr=lbl2data_repr,
        )
        

# %% ../../nbs/17_models.distillation.ipynb 52
class DTL001(DistilBertPreTrainedModel):
    use_representation,use_generation = True,False
    _tied_weights_keys = ["m_student.encoder.distilbert,m_teacher.encoder.distilbert"]
    
    def __init__(
        self,
        config,
        m_student:nn.Module,
        m_teacher:nn.Module,
        embed_sim_loss_weight:Optional[float]=1.0,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        store_attr('m_student,m_teacher')
        self.s_lw = embed_sim_loss_weight
        
        self.loss_fn = Cosine(reduce='mean')

    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        data_aug_input_ids:Optional[torch.Tensor]=None,
        data_aug_attention_mask:Optional[torch.Tensor]=None,
        **kwargs
    ):
        student_o = self.m_student(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, **kwargs)

        loss = None
        if data_aug_input_ids is not None and student_o.loss is not None:
            with torch.no_grad(): 
                teacher_o = self.m_teacher(data_input_ids=data_aug_input_ids, data_attention_mask=data_aug_attention_mask, **kwargs)

            dloss = self.loss_fn(student_o.data_embed, data_attention_mask, teacher_o.data_embed, data_aug_attention_mask)
            dloss += self.loss_fn(student_o.lbl2data_embed, kwargs['lbl2data_attention_mask'], 
                                  teacher_o.lbl2data_embed, kwargs['lbl2data_attention_mask'])
            loss = student_o.loss + self.s_lw * dloss
            
        return XCModelOutput(
            loss=loss,
            data_repr=student_o.data_repr,
            lbl2data_repr=student_o.lbl2data_repr,
        )
        

# %% ../../nbs/17_models.distillation.ipynb 63
class DTL002(DistilBertPreTrainedModel):
    use_representation,use_generation = True,False
    _tied_weights_keys = ["m_student.encoder.distilbert"]
    
    def __init__(
        self,
        config,
        m_student:nn.Module,
        m_teacher:nn.Module,
        bsz:Optional[int]=None,
        tn_targ:Optional[int]=None,
        margin:Optional[float]=0.3,
        tau:Optional[float]=0.1,
        apply_softmax:Optional[bool]=False,
        n_negatives:Optional[int]=5,
        distil_loss_weight:Optional[float]=1.0,
        mse_loss_weight:Optional[float]=0.1,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        vself.d_lw,self.m_lw = distil_loss_weight,mse_loss_weight
        store_attr('m_student,m_teacher')
        self.mse_loss_fn = nn.MSELoss()
        self.rep_loss_fn = MultiTriplet(bsz=bsz, tn_targ=tn_targ, margin=margin, n_negatives=n_negatives, tau=tau, 
                                        apply_softmax=apply_softmax, reduce='mean')
        
    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        
        data_idx:Optional[torch.Tensor]=None,
        lbl2data_idx:Optional[torch.Tensor]=None,
        **kwargs
    ):
        student_o = self.m_student(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                                   lbl2data_idx=lbl2data_idx, **kwargs)

        loss = None
        if lbl2data_idx is not None and student_o.loss is not None:
            with torch.no_grad(): 
                teacher_o = self.m_teacher(data_idx=data_idx, lbl2data_idx=lbl2data_idx)

            dloss = self.rep_loss_fn(teacher_o.data_repr, student_o.lbl2data_repr, kwargs['lbl2data_data2ptr'], lbl2data_idx, 
                                     kwargs['plbl2data_data2ptr'], kwargs['plbl2data_idx'], **kwargs)
            
            dloss += self.rep_loss_fn(student_o.data_repr, teacher_o.lbl2data_repr, kwargs['lbl2data_data2ptr'], lbl2data_idx, 
                                      kwargs['plbl2data_data2ptr'], kwargs['plbl2data_idx'], **kwargs)

            mloss = self.mse_loss_fn(teacher_o.data_repr, student_o.data_repr) + self.mse_loss_fn(teacher_o.lbl2data_repr, student_o.lbl2data_repr)
            
            loss = student_o.loss + self.d_lw * dloss + self.m_lw * mloss
            
        return XCModelOutput(
            loss=loss,
            data_repr=student_o.data_repr,
            lbl2data_repr=student_o.lbl2data_repr,
        )
        

# %% ../../nbs/17_models.distillation.ipynb 74
class DTL003(DistilBertPreTrainedModel):
    use_representation,use_generation = True,False
    _tied_weights_keys = ["m_student.encoder.distilbert"]
    
    def __init__(
        self,
        config,
        m_student:nn.Module,
        m_teacher:nn.Module,
        bsz:Optional[int]=None,
        tn_targ:Optional[int]=None,
        margin:Optional[float]=0.3,
        tau:Optional[float]=0.1,
        apply_softmax:Optional[bool]=False,
        n_negatives:Optional[int]=5,
        teacher_data_student_label_loss_weight:Optional[float]=1.0,
        student_data_teacher_label_loss_weight:Optional[float]=1.0,
        data_mse_loss_weight:Optional[float]=0.1,
        label_mse_loss_weight:Optional[float]=0.1,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        store_attr('m_student,m_teacher')
        store_attr('teacher_data_student_label_loss_weight,student_data_teacher_label_loss_weight')
        store_attr('data_mse_loss_weight,label_mse_loss_weight')
        self.mse_loss_fn = nn.MSELoss()
        self.rep_loss_fn = MultiTriplet(bsz=bsz, tn_targ=tn_targ, margin=margin, n_negatives=n_negatives, tau=tau, 
                                        apply_softmax=apply_softmax, reduce='mean')
        
    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        
        data_idx:Optional[torch.Tensor]=None,
        lbl2data_idx:Optional[torch.Tensor]=None,
        **kwargs
    ):
        student_o = self.m_student(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                                   lbl2data_idx=lbl2data_idx, **kwargs)

        loss = None
        if lbl2data_idx is not None and student_o.loss is not None:
            with torch.no_grad(): 
                teacher_o = self.m_teacher(data_idx=data_idx, lbl2data_idx=lbl2data_idx)

            tdsl_loss = self.rep_loss_fn(teacher_o.data_repr, student_o.lbl2data_repr, kwargs['lbl2data_data2ptr'], lbl2data_idx, 
                                         kwargs['plbl2data_data2ptr'], kwargs['plbl2data_idx'], **kwargs)
            
            sdtl_loss = self.rep_loss_fn(student_o.data_fused_repr, teacher_o.lbl2data_repr, kwargs['lbl2data_data2ptr'], lbl2data_idx, 
                                         kwargs['plbl2data_data2ptr'], kwargs['plbl2data_idx'], **kwargs)

            dm_loss = self.mse_loss_fn(teacher_o.data_repr, student_o.data_fused_repr)
            lm_loss = self.mse_loss_fn(teacher_o.lbl2data_repr, student_o.lbl2data_repr)
            
            loss = student_o.loss
            loss += self.teacher_data_student_label_loss_weight * tdsl_loss
            loss += self.student_data_teacher_label_loss_weight * sdtl_loss
            loss += self.data_mse_loss_weight * dm_loss + self.label_mse_loss_weight * lm_loss
            

        return RADOutput(
            loss=loss,
            
            data_repr=student_o.data_repr,
            data_fused_repr=student_o.data_fused_repr,
            
            lbl2data_repr=student_o.lbl2data_repr,
            lbl2data_fused_repr=student_o.lbl2data_fused_repr,
        )
        

# %% ../../nbs/17_models.distillation.ipynb 84
class DTL004(DistilBertPreTrainedModel):
    use_representation,use_generation = True,False
    _tied_weights_keys = ["m_student.encoder.distilbert"]
    
    def __init__(
        self,
        config,
        m_student:nn.Module,
        m_teacher:nn.Module,
        bsz:Optional[int]=None,
        tn_targ:Optional[int]=None,
        margin:Optional[float]=0.3,
        tau:Optional[float]=0.1,
        apply_softmax:Optional[bool]=False,
        n_negatives:Optional[int]=5,
        teacher_data_student_label_loss_weight:Optional[float]=1.0,
        student_data_teacher_label_loss_weight:Optional[float]=1.0,
        data_mse_loss_weight:Optional[float]=0.1,
        label_mse_loss_weight:Optional[float]=0.1,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        store_attr('m_student,m_teacher')
        store_attr('teacher_data_student_label_loss_weight,student_data_teacher_label_loss_weight')
        store_attr('data_mse_loss_weight,label_mse_loss_weight')
        self.mse_loss_fn = nn.MSELoss()
        self.rep_loss_fn = MultiTriplet(bsz=bsz, tn_targ=tn_targ, margin=margin, n_negatives=n_negatives, tau=tau, 
                                        apply_softmax=apply_softmax, reduce='mean')
        
    def forward(
        self,
        data_input_ids:Optional[torch.Tensor]=None,
        data_attention_mask:Optional[torch.Tensor]=None,
        
        data_idx:Optional[torch.Tensor]=None,
        lbl2data_idx:Optional[torch.Tensor]=None,
        **kwargs
    ):
        student_o = self.m_student(data_input_ids=data_input_ids, data_attention_mask=data_attention_mask, 
                                   lbl2data_idx=lbl2data_idx, **kwargs)

        loss = None
        if lbl2data_idx is not None and student_o.loss is not None:
            teacher_o = self.m_teacher(data_idx=data_idx, lbl2data_idx=lbl2data_idx)

            tdsl_loss = self.rep_loss_fn(teacher_o.data_repr, student_o.lbl2data_repr, kwargs['lbl2data_data2ptr'], lbl2data_idx, 
                                         kwargs['plbl2data_data2ptr'], kwargs['plbl2data_idx'], **kwargs)
            
            sdtl_loss = self.rep_loss_fn(student_o.data_fused_repr, teacher_o.lbl2data_repr, kwargs['lbl2data_data2ptr'], lbl2data_idx, 
                                         kwargs['plbl2data_data2ptr'], kwargs['plbl2data_idx'], **kwargs)

            dm_loss = self.mse_loss_fn(teacher_o.data_repr, student_o.data_fused_repr)
            lm_loss = self.mse_loss_fn(teacher_o.lbl2data_repr, student_o.lbl2data_repr)
            
            loss = student_o.loss
            loss += self.teacher_data_student_label_loss_weight * tdsl_loss
            loss += self.student_data_teacher_label_loss_weight * sdtl_loss
            loss += self.data_mse_loss_weight * dm_loss + self.label_mse_loss_weight * lm_loss
            

        return RADOutput(
            loss=loss,
            
            data_repr=student_o.data_repr,
            data_fused_repr=student_o.data_fused_repr,
            
            lbl2data_repr=student_o.lbl2data_repr,
            lbl2data_fused_repr=student_o.lbl2data_fused_repr,
        )
        
