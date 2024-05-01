# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_block.ipynb.

# %% auto 0
__all__ = ['PARAM', 'WIKISEEALSO', 'WIKICATEGORY', 'AMAZON', 'CFGS', 'TFMS', 'XCBlock', 'prepare_batch']

# %% ../nbs/03_block.ipynb 2
import numpy as np, re, inspect
from typing import Optional, Dict
from transformers import AutoTokenizer, BatchEncoding

from fastcore.meta import *

from .data import *
from .transform import *

# %% ../nbs/03_block.ipynb 6
PARAM = {
    'info_column_names': ['identifier', 'input_text'],
    'use_tokenizer': True,
    'tokenizer': 'bert-base-cased',
    'tokenization_column': 'input_text',
    'max_sequence_length': 32,
    'pad_side': 'right',
    'inp': 'data',
    'targ': 'lbl2data',
    'ptr': 'lbl2data_data2ptr',
    'drop': True,
    'ret_t': True,
    'in_place': True,
    'collapse': True,
    'device': 'cpu',
    'tfm': 'xc',
}

# %% ../nbs/03_block.ipynb 8
WIKISEEALSO = {
    'train' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/filter_labels_train.txt',
            },
        },
        'parameters': PARAM,
    },
    'data' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/filter_labels_train.txt',
            },
            'test': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/tst_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/test.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/filter_labels_test.txt',
            },
        },
        'parameters': PARAM,
    },
    'train_meta' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/filter_labels_train.txt',
                'hlk_meta': {
                    'prefix': 'hlk',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_trn_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/hyper_link.raw.txt'
                },
            },
        },
        'parameters': PARAM,
    },
    'data_meta' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/filter_labels_train.txt',
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
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/filter_labels_test.txt',
                'hlk_meta': {
                    'prefix': 'hlk',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_tst_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/hyper_link_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiSeeAlsoTitles-320K/raw_data/hyper_link.raw.txt',
                },
            },
        },
        'parameters': PARAM,
    },
}

# %% ../nbs/03_block.ipynb 9
WIKICATEGORY = {
    'train' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/label.raw.txt',
            },
        },
        'parameters': PARAM,
    },
    'data' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/label.raw.txt',
            },
            'test': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/tst_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/test.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/label.raw.txt',
            },
        },
        'parameters': PARAM,
    },
    'train_meta' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/label.raw.txt',
                'hlk_meta': {
                    'prefix': 'hlk',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/hyper_link_trn_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/hyper_link_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/hyper_link.raw.txt'
                },
            },
        },
        'parameters': PARAM,
    },
    'data_meta' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/label.raw.txt',
                'hlk_meta': {
                    'prefix': 'hlk',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/hyper_link_trn_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/hyper_link_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/hyper_link.raw.txt'
                },
            },
            'test': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/tst_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/test.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/label.raw.txt',
                'hlk_meta': {
                    'prefix': 'hlk',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/hyper_link_tst_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/hyper_link_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-WikiTitles-500K/raw_data/hyper_link.raw.txt',
                },
            },
        },
        'parameters': PARAM,
    },
}

# %% ../nbs/03_block.ipynb 10
AMAZON = {
    'train' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/filter_labels_train.txt',
            },
        },
        'parameters': PARAM,
    },
    'data' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/filter_labels_train.txt',
            },
            'test': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/tst_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/test.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/filter_labels_test.txt',
            },
        },
        'parameters': PARAM,
    },
    'train_meta' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/filter_labels_train.txt',
                'cat_meta': {
                    'prefix': 'cat',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/category_trn_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/category_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/category.raw.txt'
                },
            },
        },
        'parameters': PARAM,
    },
    'data_meta' : {
        'path': {
            'train': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/trn_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/train.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/filter_labels_train.txt',
                'cat_meta': {
                    'prefix': 'cat',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/category_trn_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/category_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/category.raw.txt'
                },
            },
            'test': {
                'data_lbl': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/tst_X_Y.txt',
                'data_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/test.raw.txt',
                'lbl_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/label.raw.txt',
                'data_lbl_filterer': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/filter_labels_test.txt',
                'cat_meta': {
                    'prefix': 'cat',
                    'data_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/category_tst_X_Y.txt',
                    'lbl_meta': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/category_lbl_X_Y.txt',
                    'meta_info': '/home/scai/phd/aiz218323/Projects/XC_NLG/data/(mapped)LF-AmazonTitles-1.3M/raw_data/category.raw.txt',
                },
            },
        },
        'parameters': PARAM,
    },
}

# %% ../nbs/03_block.ipynb 12
CFGS = {'wiki_seealso':WIKISEEALSO, 'wiki_category':WIKICATEGORY, 'amazon':AMAZON}
TFMS = {'xc': [XCPadFeatTfm, AlignInputIdsTfm], 'ng': [NGPadFeatTfm],}

# %% ../nbs/03_block.ipynb 13
class XCBlock:

    @delegates(XCDataBlock.from_cfg)
    @classmethod
    def from_cfg(cls, cfg:str, dset:Optional[str]='wiki_seealso', bsz:Optional[int]=10, **kwargs):
        if dset not in CFGS: raise ValueError(f'Invalid `dset`({cfg})')
        cfgs = CFGS[dset]

        if cfg not in cfgs: raise ValueError(f'Invalid `cfg`({cfg})')
        cfg = cfgs[cfg] 
            
        for k in cfg['parameters']: 
            if k in kwargs and kwargs[k] is not None: cfg['parameters'][k]=kwargs.pop(k)
                
        tokz = AutoTokenizer.from_pretrained(cfg['parameters']['tokenizer'])
        cfg['parameters']['sep_tok'] = tokz.sep_token_id 
        cfg['parameters']['pad_tok'] = tokz.pad_token_id
        cfg['parameters']['batch_size'] = bsz
        
        collator = XCCollator(TfmPipeline([o(**cfg['parameters']) for o in TFMS[cfg['parameters']['tfm']]]))
        
        return XCDataBlock.from_cfg(cfg, collate_fn=collator, **kwargs)


# %% ../nbs/03_block.ipynb 33
def prepare_batch(m, b, m_args=None):
    m_kwargs = inspect.signature(m.forward).parameters
    return BatchEncoding({k:v for k,v in b.items() if k in m_kwargs or (m_args is not None and k in m_args)})
