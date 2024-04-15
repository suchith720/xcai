# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_data.ipynb.

# %% auto 0
__all__ = ['MainXCData', 'MetaXCData', 'BaseXCDataset', 'MainXCDataset', 'MetaXCDataset', 'MetaXCDatasets', 'XCDataset',
           'XCCollator', 'BaseXCDataBlock', 'XCDataBlock']

# %% ../nbs/02_data.ipynb 3
from scipy import sparse
import torch, inspect, numpy as np, pandas as pd
from IPython.display import display
from typing import Dict, Optional, Callable
from torch.utils.data import Dataset,DataLoader
from xclib.data import data_utils as du
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from fastcore.utils import *
from fastcore.meta import *
from fastcore.dispatch import *

from .core import *

# %% ../nbs/02_data.ipynb 8
class MainXCData:
    
    @classmethod
    @delegates(Info.from_txt)
    def from_file(cls, data_lbl:str, data_info:str, lbl_info:str, data_lbl_filterer:Optional[str]=None, **kwargs):
        return {
            'data_lbl': du.read_sparse_file(data_lbl),
            'data_info': Info.from_txt(data_info, **kwargs),
            'lbl_info': Info.from_txt(lbl_info, **kwargs),
            'data_lbl_filterer': Filterer.load_filter(data_lbl_filterer),
        }
    

# %% ../nbs/02_data.ipynb 10
class MetaXCData:
    
    @classmethod
    @delegates(Info.from_txt)
    def from_file(cls, data_meta:str, lbl_meta:str, meta_info:str, prefix:str, **kwargs):
        return {
            'prefix': prefix,
            'data_meta': du.read_sparse_file(data_meta),
            'lbl_meta': du.read_sparse_file(lbl_meta),
            'meta_info': Info.from_txt(meta_info, **kwargs),
        }
    

# %% ../nbs/02_data.ipynb 14
class BaseXCDataset(Dataset):
    def __init__(self):
        self.n_data, self.n_lbl, self.n_meta, self.n_samples = None, None, None, None

    def __len__(self):
        return self.n_data if self.n_data is not None else 0

    def splitter(self, valid_pct:Optional[float]=0.2, seed=None):
        if seed is not None: torch.manual_seed(seed)
        rnd_idx = list(torch.randperm(self.n_data).numpy())
        cut = int(valid_pct * self.n_data)
        train, valid = self._getitems(rnd_idx[cut:]), self._getitems(rnd_idx[:cut])
        return train, valid
    
    def _verify_info(self, info:Dict):
        if info is None: raise ValueError('`info` cannot be empty.')
        n_info = [len(v) for k,v in info.items()]
        if len(n_info) == 0 or n_info[0] == 0: raise ValueError('`info` cannot be empty.')
        if np.all([n_info[0] == o for o in n_info]) == False: 
            raise ValueError('All `data_info` fields should have equal number of elements.')
        return n_info[0]
        
    def show_data(self, n:Optional[int]=10, seed:Optional[int]=None):
        if n < 1: return
        if seed: np.random.seed(seed)
        idx = np.random.permutation(self.n_data)[:n]
        d = [self[i] for i in idx]
        df = pd.DataFrame({k:[o[k] for o in d] for k in d[0]})
        with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
            display(df)
            

# %% ../nbs/02_data.ipynb 16
class MainXCDataset(BaseXCDataset):
    def __init__(self,
                 data_info:Dict,
                 data_lbl:Optional[sparse.csr_matrix]=None,
                 lbl_info:Optional[Dict]=None,
                 data_lbl_filterer:Optional[Union[sparse.csr_matrix,np.array]]=None,
                 n_samples:Optional[int]=None,
                 **kwargs):
        super().__init__()
        store_attr('data_info,data_lbl,lbl_info,data_lbl_filterer,n_samples')
        self._verify_inputs()
        
    @classmethod
    @delegates(MainXCData.from_file)
    def from_file(cls, n_samples:Optional[int]=None, **kwargs):
        return cls(**MainXCData.from_file(**kwargs), n_samples=n_samples)
        

# %% ../nbs/02_data.ipynb 17
@patch
def _verify_inputs(cls:MainXCDataset):
    cls.n_data = cls._verify_info(cls.data_info)
    if cls.data_lbl is not None:
        if cls.n_data != cls.data_lbl.shape[0]:
            raise ValueError(f'`data_info`({cls.n_data}) should have same number of datapoints as `data_lbl`({cls.data_lbl.shape[0]})')
        cls.n_lbl = cls.data_lbl.shape[1]
        if cls.lbl_info is not None:
            n_lbl = cls._verify_info(cls.lbl_info)
            if n_lbl != cls.data_lbl.shape[1]:
                raise ValueError(f'`lbl_info`({n_lbl}) should have same number of labels as `data_lbl`({cls.data_lbl.shape[1]})')

# %% ../nbs/02_data.ipynb 18
@patch
def __getitem__(cls:MainXCDataset, idx:int):
    x = {f'data_{k}': v[idx] for k,v in cls.data_info.items()}
    if cls.n_lbl is not None:
        prefix = 'lbl2data'
        x[f'{prefix}_idx'] = cls.data_lbl[idx].indices.tolist()
        if cls.n_samples: x[f'{prefix}_idx'] = [x[f'{prefix}_idx'][i] for i in np.random.permutation(len(x[f'{prefix}_idx']))[:cls.n_samples]]
        if cls.lbl_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in cls.lbl_info.items()})
    return x
    

# %% ../nbs/02_data.ipynb 20
@patch
def _getitems(cls:MainXCDataset, idxs:List):
    return MainXCDataset(
        {k:[v[idx] for idx in idxs] for k,v in cls.data_info.items()}, 
        cls.data_lbl[idxs] if cls.data_lbl is not None else None, 
        cls.lbl_info, n_samples=cls.n_samples
    )

# %% ../nbs/02_data.ipynb 28
class MetaXCDataset(BaseXCDataset):

    def __init__(self,
                 prefix:str,
                 data_meta:sparse.csr_matrix, 
                 lbl_meta:sparse.csr_matrix, 
                 meta_info:Optional[Dict]=None, 
                 n_samples:Optional[int]=None, 
                 **kwargs):
        store_attr('prefix,data_meta,lbl_meta,meta_info,n_samples')
        self._verify_inputs()

    def _getitems(self, idxs:List):
        return MetaXCDataset(self.prefix, self.data_meta[idxs], self.lbl_meta, self.meta_info, self.n_samples)
        
    @classmethod
    @delegates(MetaXCData.from_file)
    def from_file(cls, n_samples:Optional[int]=None, **kwargs):
        return cls(**MetaXCData.from_file(**kwargs), n_samples=n_samples)

    @typedispatch
    def get_lbl_meta(self, idx:int):
        prefix = f'{self.prefix}2lbl2data'
        x = {f'{prefix}_idx': self.lbl_meta[idx].indices.tolist()}
        if self.n_samples: x[f'{prefix}_idx'] = [x[f'{prefix}_idx'][i] for i in np.random.permutation(len(x[f'{prefix}_idx']))[:self.n_samples]]
        if self.meta_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in self.meta_info.items()})
        return x
    
    @typedispatch
    def get_lbl_meta(self, idxs:List):
        prefix = f'{self.prefix}2lbl2data'
        x = {f'{prefix}_idx': [self.lbl_meta[idx].indices.tolist() for idx in idxs]}
        if self.n_samples: x[f'{prefix}_idx'] = [[o[i] for i in np.random.permutation(len(o))[:self.n_samples]] for o in x[f'{prefix}_idx']]
        if self.meta_info is not None:
            x.update({f'{prefix}_{k}':[[v[i] for i in o] for o in x[f'{prefix}_idx']] for k,v in self.meta_info.items()})
        return x
        
    def get_data_meta(self, idx:int):
        prefix = f'{self.prefix}2data'
        x = {f'{prefix}_idx': self.data_meta[idx].indices.tolist()}
        if self.n_samples: x[f'{prefix}_idx'] = [x[f'{prefix}_idx'][i] for i in np.random.permutation(len(x[f'{prefix}_idx']))[:self.n_samples]]
        if self.meta_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in self.meta_info.items()})
        return x

    def shape(self):
        return (self.n_data, self.n_lbl, self.n_meta)
        
    def show_data(self, is_lbl:Optional[bool]=False, n:Optional[int]=10, seed:Optional[int]=None):
        if n < 1: return
        if seed: np.random.seed(seed)
        idx = np.random.permutation(self.n_lbl if is_lbl else self.n_data)[:n]
        d = [self.get_lbl_meta(int(i)) for i in idx] if is_lbl else [self.get_data_meta(i) for i in idx]
        df = pd.DataFrame({k:[o[k] for o in d] for k in d[0]})
        with pd.option_context('display.max_colwidth', None):
            display(df)
    

# %% ../nbs/02_data.ipynb 30
@patch
def _verify_inputs(cls:MetaXCDataset):
    cls.n_data, cls.n_lbl, cls.n_meta = cls.data_meta.shape[0], cls.lbl_meta.shape[0], cls.data_meta.shape[1]
    if cls.lbl_meta.shape[1] != cls.n_meta:
        raise ValueError(f'`lbl_meta`({cls.lbl_meta.shape[1]}) should have same number of columns as `data_meta`({cls.n_meta}).')
    if cls.meta_info is not None:
        n_meta = cls._verify_info(cls.meta_info)
        if n_meta != cls.n_meta:
            raise ValueError(f'`meta_info`({n_meta}) should have same number of entries as number of columns of `data_meta`({cls.n_meta})')
            

# %% ../nbs/02_data.ipynb 40
class MetaXCDatasets(dict):

    def __init__(self, meta:Dict):
        super().__init__(meta)
        for o in meta: setattr(self, o, meta[o])
        

# %% ../nbs/02_data.ipynb 41
class XCDataset(BaseXCDataset):

    def __init__(self, data:MainXCDataset, **kwargs):
        super().__init__()
        self.data, self.meta = data, MetaXCDatasets({k:kwargs[k] for k in self.get_meta_args(**kwargs) if isinstance(kwargs[k], MetaXCDataset)})
        self._verify_inputs()

    def _getitems(self, idxs:List):
        return XCDataset(self.data._getitems(idxs), **{k:meta._getitems(idxs) for k,meta in self.meta.items()})

    @staticmethod
    def get_meta_args(**kwargs):
        return [k for k in kwargs if re.match(r'.*_meta$', k)]
        
    @classmethod
    @delegates(MainXCDataset.from_file)
    def from_file(cls, **kwargs):
        data = MainXCDataset.from_file(**kwargs)
        meta_kwargs = {o:kwargs.pop(o) for o in cls.get_meta_args(**kwargs)}
        meta = {k:MetaXCDataset.from_file(**v, **kwargs) for k,v in meta_kwargs.items()}
        return cls(data, **meta)

    def _verify_inputs(self):
        self.n_data, self.n_lbl = self.data.n_data, self.data.n_lbl
        if len(self.meta):
            self.n_meta = self.meta[list(self.meta.keys())[0]].n_meta
            for meta in self.meta.values():
                if meta.n_data != self.n_data: raise ValueError(f'`meta`({meta.n_data}) and `data`({self.n_data}) should have the same number of datapoints.')
                if self.n_lbl is not None and meta.n_lbl != self.n_lbl: 
                    raise ValueError(f'`meta`({meta.n_lbl}) and `data`({self.n_lbl}) should have the same number of labels.')
                if meta.n_meta != self.n_meta: raise ValueError(f'Every `meta`({meta.n_meta},{self.n_meta}) should have the same number of entries.')

    def __getitem__(self, idx:int):
        x = self.data[idx]
        if self.n_meta:
            for m in self.meta.values():
                x.update(m.get_data_meta(idx))
                if self.n_lbl: x.update(m.get_lbl_meta(x['lbl2data_idx']))
        return x

    @property
    def lbl_info(self): return self.data.lbl_info

    @property
    def lbl_dset(self): return MainXCDataset(self.data.lbl_info)

    @property
    def data_info(self): return self.data.data_info

    @property
    def data_dset(self): return MainXCDataset(self.data.data_info) 

    def one_batch(self, bsz:Optional[int]=10, seed:Optional[int]=None):
        if seed is not None: torch.manual_seed(seed)
        idxs = list(torch.randperm(len(self)).numpy())[:bsz]
        return [self[idx] for idx in idxs]
       

# %% ../nbs/02_data.ipynb 51
class XCCollator:

    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, x):
        return self.tfms(x)
        

# %% ../nbs/02_data.ipynb 68
class BaseXCDataBlock:

    @delegates(DataLoader.__init__)
    def __init__(self, 
                 dset:XCDataset, 
                 collate_fn:Callable=None,
                 **kwargs):
        self.dset, self.dl_kwargs, self.collate_fn = dset, self._get_dl_kwargs(**kwargs), collate_fn
        self.dl = DataLoader(dset, collate_fn=collate_fn, **self.dl_kwargs) if collate_fn is not None else None

    @classmethod
    @delegates(XCDataset.from_file)
    def from_file(cls, collate_fn:Callable=None, **kwargs):
        return BaseXCDataBlock(XCDataset.from_file(**kwargs), collate_fn, **kwargs)

    def __len__(self):
        return len(self.dset)

    def _get_dl_kwargs(self, **kwargs):
        dl_params = inspect.signature(DataLoader.__init__).parameters
        return {k:v for k,v in kwargs.items() if k in dl_params}

    
    def _getitems(self, idxs:List):
        return BaseXCDataBlock(self.dset._getitems(idxs), collate_fn=self.collate_fn, **self.dl_kwargs)

    @property
    def bsz(self): return self.dl.batch_size

    @bsz.setter
    def bsz(self, v):
        self.dl_kwargs['batch_size'] = v
        self.dl = DataLoader(self.dset, collate_fn=self.collate_fn, **self.dl_kwargs) if self.collate_fn is not None else None

    @property
    def data_lbl_filterer(self): return self.dset.data.data_lbl_filterer

    @data_lbl_filterer.setter
    def data_lbl_filterer(self, val): self.dset.data.data_lbl_filterer = val

    @typedispatch
    def one_batch(self):
        return next(iter(self.dl))

    @typedispatch
    def one_batch(self, bsz:int):
        self.dl_kwargs['batch_size'] = bsz
        self.dl = DataLoader(self.dset, collate_fn=self.collate_fn, **self.dl_kwargs) if self.collate_fn is not None else None
        return next(iter(self.dl))
        
        

# %% ../nbs/02_data.ipynb 69
@patch
def filterer(cls:BaseXCDataBlock, train:'BaseXCDataBlock', valid:'BaseXCDataBlock', fld:Optional[str]='identifier'):
    train_info, valid_info, lbl_info = train.dset.data.data_info, valid.dset.data.data_info, train.dset.data.lbl_info
    if fld not in train_info: raise ValueError(f'`{fld}` not in `data_info`')
        
    train.data_lbl_filterer, valid_filterer = Filterer.generate(train_info[fld], valid_info[fld], lbl_info[fld], 
                                                                train.dset.data.data_lbl, valid.dset.data.data_lbl)
    _, valid_filterer, idx = Filterer.prune(valid.dset.data.data_lbl, valid_filterer)
    
    valid = valid._getitems(idx)
    valid.data_lbl_filterer = valid_filterer
    
    return train, valid

@patch
def splitter(cls:BaseXCDataBlock, valid_pct:Optional[float]=0.2, seed=None):
    if seed is not None: torch.manual_seed(seed)
    rnd_idx = list(torch.randperm(len(cls)).numpy())
    cut = int(valid_pct * len(cls))
    train, valid = cls._getitems(rnd_idx[cut:]), cls._getitems(rnd_idx[:cut])
    if cls.data_lbl_filterer is None: return train, valid
    else: return cls.filterer(train, valid)
        

# %% ../nbs/02_data.ipynb 78
class XCDataBlock:

    def __init__(self, train:BaseXCDataBlock=None, valid:BaseXCDataBlock=None, test:BaseXCDataBlock=None):
        self.train, self.valid, self.test = train, valid, test

    @staticmethod
    def load_cfg(fname):
        with open(fname, 'r') as f: return json.load(f)

    @property
    def lbl_info(self): return self.train.dset.data.lbl_info

    @property
    def lbl_dset(self): return MainXCDataset(self.train.dset.data.lbl_info)

    @property
    def n_lbl(self): return self.train.dset.n_lbl

    @property
    def collator(self): return self.train.collate_fn
        
    @classmethod
    def from_cfg(cls, 
                 cfg:Union[str,Dict],
                 collate_fn:Optional[Callable]=None,
                 valid_pct:Optional[float]=0.2,
                 seed=None):
        if isinstance(cfg, str): cfg = cls.load_cfg(cfg)
        blks = {o:BaseXCDataBlock.from_file(**cfg['path'][o], **cfg['parameters'], collate_fn=collate_fn) for o in ['train', 'valid', 'test'] if o in cfg['path']}
        if 'valid' not in blks: blks['train'], blks['valid'] = blks['train'].splitter(valid_pct, seed=seed)
        return cls(**blks)
        
