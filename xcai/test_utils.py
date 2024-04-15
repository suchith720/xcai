# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_test_utils.ipynb.

# %% auto 0
__all__ = ['PARAM', 'TRAIN_CFG', 'DATA_CFG', 'TRAIN_META_CFG', 'DATA_META_CFG', 'CFGS', 'XCBlock', 'prepare_batch']

# %% ../nbs/03_test_utils.ipynb 2
import numpy as np, re, inspect
from .data import *
from .transform import *
from fastcore.meta import *
from typing import Optional, Dict
from transformers import AutoTokenizer, BatchEncoding

# %% ../nbs/03_test_utils.ipynb 5
PARAM = {
    'cols': ['identifier', 'input_text'],
    'use_tokz': True,
    'tokz': 'bert-base-cased',
    'fld': 'input_text',
    'max_len': 32,
    'pad_side': 'right',
    'inp': 'data',
    'targ': 'lbl2data',
    'ptr': 'lbl2data_data2ptr',
    'drop': True,
    'ret_t': True,
    'in_place': True,
    'collapse': True,
    'device': 'cpu',
}

# %% ../nbs/03_test_utils.ipynb 6
TRAIN_CFG = {
    'path': {
        'train': {
            'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
            'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
            'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
        },
    },
    'parameters': PARAM,
}

# %% ../nbs/03_test_utils.ipynb 7
DATA_CFG = {
    'path': {
        'train': {
            'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
            'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
            'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
        },
        'test': {
            'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/tst_X_Y.txt',
            'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/test.raw.txt',
            'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
        },
    },
    'parameters': PARAM,
}

# %% ../nbs/03_test_utils.ipynb 8
TRAIN_META_CFG = {
    'path': {
        'train': {
            'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
            'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
            'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
            'hlk_meta': {
                'prefix': 'hlk',
                'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_trn_X_Y.txt',
                'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_lbl_X_Y.txt',
                'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/hyper_link.raw.txt'
            },
        },
    },
    'parameters': PARAM,
}

# %% ../nbs/03_test_utils.ipynb 9
DATA_META_CFG = {
    'path': {
        'train': {
            'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
            'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
            'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
            'hlk_meta': {
                'prefix': 'hlk',
                'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_trn_X_Y.txt',
                'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_lbl_X_Y.txt',
                'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/hyper_link.raw.txt'
            },
        },
        'test': {
            'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/tst_X_Y.txt',
            'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/test.raw.txt',
            'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
            'hlk_meta': {
                'prefix': 'hlk',
                'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_tst_X_Y.txt',
                'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_lbl_X_Y.txt',
                'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/hyper_link.raw.txt',
            },
        },
    },
    'parameters': PARAM,
}

# %% ../nbs/03_test_utils.ipynb 11
CFGS = {'train':TRAIN_CFG, 'data':DATA_CFG, 'data_meta':DATA_META_CFG, 'train_meta':TRAIN_META_CFG}

# %% ../nbs/03_test_utils.ipynb 12
class XCBlock:

    @delegates(XCDataBlock.from_cfg)
    @classmethod
    def from_cfg(cls, cfg:str, bsz:Optional[int]=10, **kwargs):
        if cfg in CFGS: cfg = CFGS[cfg]
        else: raise ValueError(f"Invalid `cfg` value {cfg}, it should be one of ['train', 'data', 'train_meta', 'data_meta']")
        for k in cfg['parameters']: 
            if k in kwargs and kwargs[k] is not None: cfg['parameters'][k]=kwargs.pop(k)
        tokz = AutoTokenizer.from_pretrained(cfg['parameters']['tokz'])
        cfg['parameters']['sep_tok'], cfg['parameters']['pad_tok'], cfg['parameters']['batch_size'] = tokz.sep_token_id, tokz.pad_token_id, bsz
        collator = XCCollator(TfmPipeline([XCPadFeatTfm(**cfg['parameters']), AlignInputIdsTfm(**cfg['parameters'])]))
        return XCDataBlock.from_cfg(cfg, collate_fn=collator, **kwargs)


# %% ../nbs/03_test_utils.ipynb 20
def prepare_batch(m, b, m_args=None):
    m_kwargs = inspect.signature(m.forward).parameters
    return BatchEncoding({k:v for k,v in b.items() if k in m_kwargs or (m_args is not None and k in m_args)})
