# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/35_sdata.ipynb.

# %% auto 0
__all__ = ['identity_collate_fn', 'SMainXCDataset', 'SMetaXCDataset', 'SXCDataset']

# %% ../nbs/35_sdata.ipynb 3
import torch

from .data import MainXCData, MetaXCData
from .data import BaseXCDataset, MainXCDataset, MetaXCDataset, MetaXCDatasets

from fastcore.utils import *
from fastcore.meta import *

# %% ../nbs/35_sdata.ipynb 9
def identity_collate_fn(batch): return batch

# %% ../nbs/35_sdata.ipynb 12
class SMainXCDataset(MainXCDataset):

    def __init__(
        self,
        n_slbl_samples:Optional[int]=1,
        main_oversample:Optional[bool]=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        store_attr('n_slbl_samples,main_oversample')

    def __getitems__(self, idxs:List):
        x = {'data_idx': torch.tensor(idxs, dtype=torch.int64)}
        x.update(self.get_info('data', idxs, self.data_info, self.data_info_keys))
        if self.n_lbl is not None:
            prefix = 'lbl2data'
            o = self.extract_items(prefix, self.curr_data_lbl, idxs, self.n_lbl_samples, self.n_slbl_samples, self.main_oversample, 
                                   self.lbl_info, self.lbl_info_keys)
            x.update(o)
        return x

    @classmethod
    @delegates(MainXCData.from_file)
    def from_file(
        cls, 
        n_lbl_samples:Optional[int]=None,
        data_info_keys:Optional[List]=None,
        lbl_info_keys:Optional[List]=None,
        n_slbl_samples:Optional[int]=1,
        main_oversample:Optional[bool]=False,
        **kwargs
    ):
        return cls(**MainXCData.from_file(**kwargs), n_lbl_samples=n_lbl_samples, data_info_keys=data_info_keys, 
                   lbl_info_keys=lbl_info_keys, n_slbl_samples=n_slbl_samples, main_oversample=main_oversample)
    

# %% ../nbs/35_sdata.ipynb 13
@patch
def _getitems(cls:MainXCDataset, idxs:List):
    return MainXCDataset(
        {k:[v[idx] for idx in idxs] for k,v in cls.data_info.items()}, 
        data_lbl=cls.data_lbl[idxs] if cls.data_lbl is not None else None, 
        lbl_info=cls.lbl_info, 
        data_lbl_filterer=Filterer.sample(cls.data_lbl_filterer, sz=cls.data_lbl.shape, idx=idxs) if cls.data_lbl_filterer is not None else None,
        n_lbl_samples=cls.n_lbl_samples,
        data_info_keys=cls.data_info_keys,
        lbl_info_keys=cls.lbl_info_keys,
        n_slbl_samples=cls.n_slbl_samples,
        main_oversample=cls.main_oversample,
    )
    

# %% ../nbs/35_sdata.ipynb 24
class SMetaXCDataset(MetaXCDataset):

    def __init__(
        self,
        n_sdata_meta_samples:Optional[int]=1,
        n_slbl_meta_samples:Optional[int]=1,
        meta_oversample:Optional[bool]=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        store_attr('n_sdata_meta_samples,n_slbl_meta_samples,meta_oversample')

    def _getitems(self, idxs:List):
        return SMetaXCDataset(prefix=self.prefix, data_meta=self.data_meta[idxs], lbl_meta=self.lbl_meta, meta_info=self.meta_info, 
                              n_data_meta_samples=self.n_data_meta_samples, n_lbl_meta_samples=self.n_lbl_meta_samples, 
                              meta_info_keys=self.meta_info_keys, n_sdata_meta_samples=self.n_sdata_meta_samples, 
                              n_slbl_meta_samples=self.n_slbl_meta_samples, meta_oversample=self.meta_oversample)

    def _sample_meta_items(self, idxs:List):
        assert max(idxs) < self.n_meta, f"indices should be less than {self.n_meta}"
        meta_info = {k: [v[i] for i in idxs] for k,v in self.meta_info.items()}
        return SMetaXCDataset(prefix=self.prefix, data_meta=self.data_meta[:, idxs], lbl_meta=self.lbl_meta[:, idxs], meta_info=meta_info, 
                              n_data_meta_samples=self.n_data_meta_samples, n_lbl_meta_samples=self.n_lbl_meta_samples,
                              meta_info_keys=self.meta_info_keys, n_sdata_meta_samples=self.n_sdata_meta_samples, 
                              n_slbl_meta_samples=self.n_slbl_meta_samples, meta_oversample=self.meta_oversample)
        
    @classmethod
    @delegates(MetaXCData.from_file)
    def from_file(
        cls, 
        n_data_meta_samples:Optional[int]=None, 
        n_lbl_meta_samples:Optional[int]=None, 
        meta_info_keys:Optional[List]=None,
        n_sdata_meta_samples:Optional[int]=1,
        n_slbl_meta_samples:Optional[int]=1,
        meta_oversample:Optional[bool]=False,
        **kwargs
    ):
        return cls(**MetaXCData.from_file(**kwargs), n_data_meta_samples=n_data_meta_samples, n_lbl_meta_samples=n_lbl_meta_samples, 
                   meta_info_keys=meta_info_keys, n_sdata_meta_samples=n_sdata_meta_samples, n_slbl_meta_samples=n_slbl_meta_samples,
                   meta_oversample=meta_oversample)

    def get_data_meta(self, idxs:List):
        x, prefix = dict(), f'{self.prefix}2data'
        o = self.extract_items(prefix, self.curr_data_meta, idxs, self.n_data_meta_samples, self.n_sdata_meta_samples, 
                               self.meta_oversample, self.meta_info, self.meta_info_keys)
        x.update(o)
        return x
        
    def get_lbl_meta(self, idxs:List):
        x, prefix = dict(), f'{self.prefix}2lbl'
        o = self.extract_items(prefix, self.curr_lbl_meta, idxs, self.n_lbl_meta_samples, self.n_slbl_meta_samples, 
                               self.meta_oversample, self.meta_info, self.meta_info_keys)
        x.update(o)
        return x
        

# %% ../nbs/35_sdata.ipynb 31
class SXCDataset(BaseXCDataset):

    def __init__(self, data:SMainXCDataset, **kwargs):
        super().__init__()
        self.data, self.meta = data, MetaXCDatasets({k:kwargs[k] for k in self.get_meta_args(**kwargs) if isinstance(kwargs[k], SMetaXCDataset)})
        self._verify_inputs()

    def _getitems(self, idxs:List):
        return SXCDataset(self.data._getitems(idxs), **{k:meta._getitems(idxs) for k,meta in self.meta.items()})
        
    @classmethod
    @delegates(SMainXCDataset.from_file)
    def from_file(cls, **kwargs):
        data = SMainXCDataset.from_file(**kwargs)
        meta_kwargs = {o:kwargs.pop(o) for o in cls.get_meta_args(**kwargs)}
        meta = {k:SMetaXCDataset.from_file(**v, **kwargs) for k,v in meta_kwargs.items()}
        return cls(data, **meta)

    @staticmethod
    def get_meta_args(**kwargs):
        return [k for k in kwargs if re.match(r'.*_meta$', k)]

    def _verify_inputs(self):
        self.n_data, self.n_lbl = self.data.n_data, self.data.n_lbl
        if len(self.meta):
            self.n_meta = len(self.meta)
            for meta in self.meta.values():
                if meta.n_data != self.n_data: 
                    raise ValueError(f'`meta`({meta.n_data}) and `data`({self.n_data}) should have the same number of datapoints.')
                if self.n_lbl is not None and meta.n_lbl != self.n_lbl: 
                    raise ValueError(f'`meta`({meta.n_lbl}) and `data`({self.n_lbl}) should have the same number of labels.')

    def __getitems__(self, idxs:List):
        x = self.data.__getitems__(idxs)
        if self.n_meta:
            for meta in self.meta.values():
                x.update(meta.get_data_meta(idxs))
                if self.n_lbl:
                    z = meta.get_lbl_meta(x['lbl2data_idx'])
                    z[f'{meta.prefix}2lbl_data2ptr'] = torch.tensor([o.sum() for o in z[f'{meta.prefix}2lbl_lbl2ptr'].split_with_sizes(z[f'lbl2data_data2ptr'].tolist())])
        return x

    @property
    def lbl_info(self): return self.data.lbl_info

    @property
    def lbl_dset(self): return SMainXCDataset(self.data.lbl_info)

    @property
    def data_info(self): return self.data.data_info

    @property
    def data_dset(self): return SMainXCDataset(self.data.data_info) 

    def one_batch(self, bsz:Optional[int]=10, seed:Optional[int]=None):
        if seed is not None: torch.manual_seed(seed)
        idxs = list(torch.randperm(len(self)).numpy())[:bsz]
        return [self[idx] for idx in idxs]
       
