"""Datasets and collators for Extreme Classification"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_data.ipynb.

# %% auto 0
__all__ = ['MainXCData', 'MetaXCData', 'BaseXCDataset', 'MainXCDataset', 'MetaXCDataset', 'MetaXCDatasets', 'XCDataset',
           'XCCollator', 'BaseXCDataBlock', 'XCDataBlock']

# %% ../nbs/02_data.ipynb 3
from scipy import sparse
from tqdm.auto import tqdm
import torch, inspect, numpy as np, pandas as pd, torch.nn.functional as F
from IPython.display import display
from typing import Dict, Optional, Callable
from torch.utils.data import Dataset,DataLoader
from xclib.data import data_utils as du
from xclib.utils.sparse import retain_topk
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from fastcore.utils import *
from fastcore.meta import *
from fastcore.dispatch import *

from .core import *

# %% ../nbs/02_data.ipynb 8
def _read_sparse_file(fname:str):
    if fname.endswith('.txt'): return du.read_sparse_file(fname)
    elif fname.endswith('.npz'): return sparse.load_npz(fname)
    else: raise ValueError(f'Invalid file extension : {fname}')
    

# %% ../nbs/02_data.ipynb 9
class MainXCData:
    
    @classmethod
    @delegates(Info.from_txt)
    def from_file(cls, data_lbl:str, data_info:str, lbl_info:str, data_lbl_filterer:Optional[str]=None, **kwargs):
        return {
            'data_lbl': _read_sparse_file(data_lbl),
            'data_info': Info.from_txt(data_info, **kwargs),
            'lbl_info': Info.from_txt(lbl_info, **kwargs),
            'data_lbl_filterer': Filterer.load_filter(data_lbl_filterer),
        }
    

# %% ../nbs/02_data.ipynb 11
class MetaXCData:
    
    @classmethod
    @delegates(Info.from_txt)
    def from_file(cls, data_meta:str, lbl_meta:str, meta_info:str, prefix:str, **kwargs):
        return {
            'prefix': prefix,
            'data_meta': _read_sparse_file(data_meta),
            'lbl_meta': _read_sparse_file(lbl_meta),
            'meta_info': Info.from_txt(meta_info, **kwargs),
        }
    

# %% ../nbs/02_data.ipynb 15
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

    def sample(self, pct:Optional[float]=0.2, n:Optional[int]=None, seed=None):
        if seed is not None: torch.manual_seed(seed)
        rnd_idx = list(torch.randperm(self.n_data).numpy())
        cut = int(pct * self.n_data) if n is None else max(1, n)
        return self._getitems(rnd_idx[:cut])
        
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

    def prune_data_lbl(self, data_lbl:sparse.csr_matrix, data_repr:torch.Tensor, lbl_repr:torch.Tensor, batch_size:Optional[int]=64, 
                       thresh:Optional[float]=0.1, topk:Optional[int]=None):
        data_repr,lbl_repr = F.normalize(data_repr, dim=1), F.normalize(lbl_repr, dim=1)
        curr_data_lbl = data_lbl.copy()
        rows, cols = data_lbl.nonzero()
        dl = DataLoader(list(zip(rows, cols)), batch_size=batch_size, shuffle=False)
        score = None
        for b in tqdm(dl, total=len(dl)): 
            sc = data_repr[b[0]].unsqueeze(1)@lbl_repr[b[1]].unsqueeze(2)
            sc = sc.squeeze()
            sc = torch.where(sc < thresh, 0, sc)
            score = sc if score is None else torch.hstack([score, sc])
        curr_data_lbl.data[:] = score
        curr_data_lbl.eliminate_zeros()
        if topk is not None: 
            curr_data_lbl = retain_topk(curr_data_lbl, k=topk)
        return curr_data_lbl
            

# %% ../nbs/02_data.ipynb 17
class MainXCDataset(BaseXCDataset):
    def __init__(self,
                 data_info:Dict,
                 data_lbl:Optional[sparse.csr_matrix]=None,
                 lbl_info:Optional[Dict]=None,
                 data_lbl_filterer:Optional[Union[sparse.csr_matrix,np.array]]=None,
                 n_lbl_samples:Optional[int]=None,
                 data_info_keys:Optional[List]=None,
                 lbl_info_keys:Optional[List]=None,
                 **kwargs):
        super().__init__()
        store_attr('data_info,data_lbl,lbl_info,data_lbl_filterer,n_lbl_samples,data_info_keys,lbl_info_keys')
        self.curr_data_lbl = None
        
        self._verify_inputs()
        self._store_indices()
        
    @classmethod
    @delegates(MainXCData.from_file)
    def from_file(cls, n_lbl_samples:Optional[int]=None, lbl_info_keys:Optional[List]=None, **kwargs):
        return cls(**MainXCData.from_file(**kwargs), n_lbl_samples=n_lbl_samples, lbl_info_keys=lbl_info_keys)
        

# %% ../nbs/02_data.ipynb 18
@patch
def _store_indices(cls:MainXCDataset):
    if cls.data_lbl is not None: cls.curr_data_lbl = [o.indices.tolist() for o in cls.data_lbl]


# %% ../nbs/02_data.ipynb 19
@patch
def _verify_inputs(cls:MainXCDataset):
    cls.n_data = cls._verify_info(cls.data_info)
    if cls.data_info_keys is None: cls.data_info_keys = list(cls.data_info.keys())
    if cls.data_lbl is not None:
        if cls.n_data != cls.data_lbl.shape[0]:
            raise ValueError(f'`data_info`({cls.n_data}) should have same number of datapoints as `data_lbl`({cls.data_lbl.shape[0]})')
        cls.n_lbl = cls.data_lbl.shape[1]
        if cls.lbl_info is not None:
            n_lbl = cls._verify_info(cls.lbl_info)
            if n_lbl != cls.data_lbl.shape[1]:
                raise ValueError(f'`lbl_info`({n_lbl}) should have same number of labels as `data_lbl`({cls.data_lbl.shape[1]})')
            if cls.lbl_info_keys is None: cls.lbl_info_keys = list(cls.lbl_info.keys())
                

# %% ../nbs/02_data.ipynb 20
@patch
def __getitem__(cls:MainXCDataset, idx:int):
    x = {f'data_{k}': v[idx] for k,v in cls.data_info.items() if k in cls.data_info_keys}
    x['data_idx'] = idx
    if cls.n_lbl is not None:
        prefix = 'lbl2data'
        x[f'{prefix}_idx'] = cls.curr_data_lbl[idx]
        if cls.n_lbl_samples: x[f'{prefix}_idx'] = [x[f'{prefix}_idx'][i] for i in np.random.permutation(len(x[f'{prefix}_idx']))[:cls.n_lbl_samples]]
        if cls.lbl_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in cls.lbl_info.items() if k in cls.lbl_info_keys})
    return x
    

# %% ../nbs/02_data.ipynb 22
@patch
def _getitems(cls:MainXCDataset, idxs:List):
    return MainXCDataset(
        {k:[v[idx] for idx in idxs] for k,v in cls.data_info.items()}, 
        cls.data_lbl[idxs] if cls.data_lbl is not None else None, cls.lbl_info, 
        Filterer.sample(cls.data_lbl_filterer, sz=cls.data_lbl.shape, idx=idxs) if cls.data_lbl_filterer is not None else None,
        n_lbl_samples=cls.n_lbl_samples,
        data_info_keys=cls.data_info_keys,
        lbl_info_keys=cls.lbl_info_keys,
    )

# %% ../nbs/02_data.ipynb 31
class MetaXCDataset(BaseXCDataset):

    def __init__(self,
                 prefix:str,
                 data_meta:sparse.csr_matrix, 
                 lbl_meta:sparse.csr_matrix, 
                 meta_info:Optional[Dict]=None, 
                 n_data_meta_samples:Optional[int]=None,
                 n_lbl_meta_samples:Optional[int]=None,
                 meta_info_keys:Optional[List]=None,
                 **kwargs):
        store_attr('prefix,data_meta,lbl_meta,meta_info,n_data_meta_samples,n_lbl_meta_samples,meta_info_keys')
        self.curr_data_meta,self.curr_lbl_meta = None,None
        self._verify_inputs()
        self._store_indices()

    def prune_data_meta(self, data_repr:torch.Tensor, meta_repr:torch.Tensor, batch_size:Optional[int]=64, thresh:Optional[float]=0.0, 
                        topk:Optional[int]=None):
        data_meta = self.prune_data_lbl(self.data_meta, data_repr, meta_repr, batch_size, thresh, topk)
        self.curr_data_meta = [o.indices.tolist() for o in data_meta]

    def prune_lbl_meta(self, lbl_repr:torch.Tensor, meta_repr:torch.Tensor, batch_size:Optional[int]=64, thresh:Optional[float]=0.0, 
                       topk:Optional[int]=None):
        lbl_meta = self.prune_data_lbl(self.lbl_meta, lbl_repr, meta_repr, batch_size, thresh, topk)
        self.curr_lbl_meta = [o.indices.tolist() for o in lbl_meta]

    def _store_indices(self):
        if self.data_meta is not None: self.curr_data_meta = [o.indices.tolist() for o in self.data_meta]
        if self.lbl_meta is not None: self.curr_lbl_meta = [o.indices.tolist() for o in self.lbl_meta]

    def update_meta_matrix(self, data_meta:sparse.csr_matrix, lbl_meta:sparse.csr_matrix):
        self.data_meta, self.lbl_meta = data_meta, lbl_meta
        self._store_indices()
        
    def _getitems(self, idxs:List):
        return MetaXCDataset(self.prefix, self.data_meta[idxs], self.lbl_meta, self.meta_info, 
                             self.n_data_meta_samples, self.n_lbl_meta_samples, self.meta_info_keys)
        
    @classmethod
    @delegates(MetaXCData.from_file)
    def from_file(cls, n_data_meta_samples:Optional[int]=None, n_lbl_meta_samples:Optional[int]=None, meta_info_keys:Optional[List]=None, **kwargs):
        return cls(**MetaXCData.from_file(**kwargs), n_data_meta_samples=n_data_meta_samples, 
                   n_lbl_meta_samples=n_lbl_meta_samples, meta_info_keys=meta_info_keys)

    @typedispatch
    def get_lbl_meta(self, idx:int):
        prefix = f'{self.prefix}2lbl2data'
        x = {f'{prefix}_idx': self.curr_lbl_meta[idx]}
        if self.n_lbl_meta_samples: x[f'{prefix}_idx'] = [x[f'{prefix}_idx'][i] for i in np.random.permutation(len(x[f'{prefix}_idx']))[:self.n_lbl_meta_samples]]
        if self.meta_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in self.meta_info.items() if k in self.meta_info_keys})
        return x
    
    @typedispatch
    def get_lbl_meta(self, idxs:List):
        prefix = f'{self.prefix}2lbl2data'
        x = {f'{prefix}_idx': [self.curr_lbl_meta[idx] for idx in idxs]}
        if self.n_lbl_meta_samples: x[f'{prefix}_idx'] = [[o[i] for i in np.random.permutation(len(o))[:self.n_lbl_meta_samples]] for o in x[f'{prefix}_idx']]
        if self.meta_info is not None:
            x.update({f'{prefix}_{k}':[[v[i] for i in o] for o in x[f'{prefix}_idx']] for k,v in self.meta_info.items() if k in self.meta_info_keys})
        return x
        
    def get_data_meta(self, idx:int):
        prefix = f'{self.prefix}2data'
        x = {f'{prefix}_idx': self.curr_data_meta[idx]}
        if self.n_data_meta_samples: x[f'{prefix}_idx'] = [x[f'{prefix}_idx'][i] for i in np.random.permutation(len(x[f'{prefix}_idx']))[:self.n_data_meta_samples]]
        if self.meta_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in self.meta_info.items() if k in self.meta_info_keys})
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
    

# %% ../nbs/02_data.ipynb 33
@patch
def _verify_inputs(cls:MetaXCDataset):
    cls.n_data,cls.n_meta = cls.data_meta.shape[0],cls.data_meta.shape[1]
    
    if cls.lbl_meta is not None:
        cls.n_lbl = cls.lbl_meta.shape[0]
        if cls.lbl_meta.shape[1] != cls.n_meta:
            raise ValueError(f'`lbl_meta`({cls.lbl_meta.shape[1]}) should have same number of columns as `data_meta`({cls.n_meta}).')

    if cls.meta_info is not None:
        n_meta = cls._verify_info(cls.meta_info)
        if n_meta != cls.n_meta:
            raise ValueError(f'`meta_info`({n_meta}) should have same number of entries as number of columns of `data_meta`({cls.n_meta})')
        if cls.meta_info_keys is None: cls.meta_info_keys = list(cls.meta_info.keys())
            

# %% ../nbs/02_data.ipynb 44
class MetaXCDatasets(dict):

    def __init__(self, meta:Dict):
        super().__init__(meta)
        for o in meta: setattr(self, o, meta[o])
        

# %% ../nbs/02_data.ipynb 45
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

    def _retain_randk(self, matrix:sparse.csr_matrix, topk:Optional[int]=3):
        data, indices, indptr = [], [], np.zeros_like(matrix.indptr)
        for i,row in tqdm(enumerate(matrix), total=matrix.shape[0]):
            if row.nnz > 0:
                idx = np.random.randint(row.nnz, size=topk)
                ind, d = row.indices[idx], row.data[idx]    
            else:
                ind, d = np.arange(topk), np.zeros(topk)
                
            indptr[i+1] = indptr[i] + topk
            indices.append(ind); data.append(d)
        data = np.hstack(data)
        indices = np.hstack(indices)
        o = sparse.csr_matrix((data, indices, indptr), shape=matrix.shape, dtype=matrix.dtype)
        o.sort_indices()
        return o

    def _remove_data(self, meta_1:sparse.csr_matrix, meta_2:sparse.csr_matrix, pct:Optional[float]=0.3):
        n_data = min(len(meta_1.data),len(meta_2.data))
        n = int(n_data * pct)
        idx = np.random.permutation(n_data)
        idx_1,idx_2 = idx[:n], idx[n:]
        meta_1.data[idx_1] = 0; meta_1.eliminate_zeros()
        meta_2.data[idx_2] = 0; meta_2.eliminate_zeros()

    def _mix_meta_matrix(self, meta_1:sparse.csr_matrix, meta_2:sparse.csr_matrix, pct:Optional[float]=0.3, k:Optional[int]=3):
        meta_1 = self._retain_randk(meta_1, topk=k)
        meta_2 = self._retain_randk(meta_2, topk=k)
        self._remove_data(meta_1, meta_2, pct)
        return meta_1 + meta_2

    def mix_meta_dataset(self, meta_1:str, meta_2:str, pct:Optional[float]=0.3, k:Optional[int]=3):
        if pct < 1:
            meta_info = self.meta[f'{meta_1}_meta'].meta_info
            data_meta = self._mix_meta_matrix(self.meta[f'{meta_1}_meta'].data_meta, self.meta[f'{meta_2}_meta'].data_meta, pct=pct, k=k)
            lbl_meta = self._mix_meta_matrix(self.meta[f'{meta_1}_meta'].lbl_meta, self.meta[f'{meta_2}_meta'].lbl_meta, pct=pct, k=k)
            self.meta['hyb_meta'] = MetaXCDataset('hyb', data_meta, lbl_meta, meta_info, 
                                                  n_data_meta_samples=self.meta[f'{meta_2}_meta'].n_data_meta_samples,
                                                  n_lbl_meta_samples=self.meta[f'{meta_2}_meta'].n_lbl_meta_samples, 
                                                  meta_info_keys=self.meta[f'{meta_2}_meta'].meta_info_keys)
        else:
            self.meta['hyb_meta'] = MetaXCDataset('hyb', self.meta[f'{meta_2}_meta'].data_meta, 
                                                  self.meta[f'{meta_2}_meta'].lbl_meta, 
                                                  self.meta[f'{meta_2}_meta'].meta_info, 
                                                  n_data_meta_samples=self.meta[f'{meta_2}_meta'].n_data_meta_samples,
                                                  n_lbl_meta_samples=self.meta[f'{meta_2}_meta'].n_lbl_meta_samples, 
                                                  meta_info_keys=self.meta[f'{meta_2}_meta'].meta_info_keys)
       

# %% ../nbs/02_data.ipynb 57
class XCCollator:

    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, x):
        return self.tfms(x)
        

# %% ../nbs/02_data.ipynb 74
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
        
        

# %% ../nbs/02_data.ipynb 75
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

@patch
def sample(cls:BaseXCDataBlock, pct:Optional[float]=0.2, n:Optional[int]=None, seed=None):
    if seed is not None: torch.manual_seed(seed)
    rnd_idx = list(torch.randperm(len(cls)).numpy())
    cut = int(pct * len(cls)) if n is None else max(1, n)
    return cls._getitems(rnd_idx[:cut])
    

# %% ../nbs/02_data.ipynb 85
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
        # if 'valid' not in blks: blks['train'], blks['valid'] = blks['train'].splitter(valid_pct, seed=seed)
        return cls(**blks)
        
