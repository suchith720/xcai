# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/17_models.distillation.ipynb.

# %% auto 0
__all__ = ['DTL001']

# %% ../../nbs/17_models.distillation.ipynb 2
import torch, numpy as np
import torch.nn as nn
from typing import Optional

from ..core import store_attr
from ..losses import Cosine
from .PPP0XX import XCModelOutput

# %% ../../nbs/17_models.distillation.ipynb 9
class DTL001(nn.Module):

    def __init__(
        self,
        m_student:nn.Module,
        m_teacher:nn.Module,
        embed_sim_loss_weight:Optional[float]=1.0,
    ):
        super().__init__()
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
        
