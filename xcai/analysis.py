# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/16_analysis.ipynb.

# %% auto 0
__all__ = ['pointwise_eval', 'equal_volume_split', 'get_decile_stats', 'barplot', 'decile_plot', 'get_pred_dset',
           'get_pred_sparse', 'load_pred_sparse', 'get_output', 'load_prediction_and_block', 'PredictionBlock',
           'TextDataset', 'CompareDataset', 'Indices']

# %% ../nbs/16_analysis.ipynb 2
import os,torch, torch.multiprocessing as mp, pickle, numpy as np, re
from typing import Optional, Dict, List, Tuple
from torch.utils.data import Dataset
from scipy import sparse
from termcolor import colored, COLORS
import matplotlib.pyplot as plt

from fastcore.dispatch import *

from .basics import *
from .data import *
from .sdata import *
from .learner import XCPredictionOutput

import xclib.utils.sparse as xc_sparse
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as du 

from IPython.display import HTML

# %% ../nbs/16_analysis.ipynb 8
def pointwise_eval(pred_lbl:sparse.csr_matrix, data_lbl:sparse.csr_matrix, data_lbl_filterer:Optional[np.ndarray]=None,
                   topk:Optional[int]=5, metric:Optional[str]='P', return_type:Optional[str]='M'):
    
    if data_lbl_filterer is not None:
        pred_lbl = Filterer.apply(pred_lbl, data_lbl_filterer)
        data_lbl = Filterer.apply(data_lbl, data_lbl_filterer)
        
    pred_lbl = xc_sparse.retain_topk(pred_lbl, k=topk)
        
    scores = pred_lbl.multiply(data_lbl)
    scores.data[:] = 1
        
    if metric == 'P':
        scores = scores.multiply(1/(topk * data_lbl.shape[0]))
    elif metric == 'R':
        d = data_lbl.getnnz(axis=1) * data_lbl.shape[0]
        scores = scores.multiply(1/d.reshape(-1,1))
    elif metric == 'FN':
        scores = data_lbl - scores
        scores.eliminate_zeros()
        lbl_cnt = data_lbl.getnnz(axis=0)
        scores = scores.multiply(1/(lbl_cnt * data_lbl.shape[1]))

    if return_type == 'M': return scores
    elif return_type == 'L': return np.ravel(scores.sum(axis=0))
    else: return np.ravel(scores.sum(axis=1))
    

# %% ../nbs/16_analysis.ipynb 10
def equal_volume_split(data_lbl:sparse.csr_matrix, n_split:int):
    lbl_cnt = data_lbl.getnnz(axis=0)
    lbl_idx = np.argsort(-lbl_cnt)
    thresh = lbl_cnt.sum()/n_split
    
    splits,split,cnt = [],[],0
    for idx in lbl_idx:
        cnt += lbl_cnt[idx]
        split.append(idx)

        if cnt > thresh: 
            splits.append(split)
            split,cnt = [],0

    if len(split): splits.append(split)
    
    if len(splits) != n_split: raise ValueError(f'Number of splits created less than {n_split}.')
    splits.append(lbl_idx.tolist())
    
    lbl_cnt = lbl_cnt.astype(np.float32)
    lbl_cnt[lbl_cnt == 0] = np.nan
    info = [f'{i+1}\n{len(split)//1000}K\n{np.nanmean(lbl_cnt[split]):.2f}' for i,split in enumerate(splits)]
    return splits,info
    

# %% ../nbs/16_analysis.ipynb 11
def get_decile_stats(pred_lbl:sparse.csr_matrix, data_lbl:sparse.csr_matrix, data_lbl_filterer:np.ndarray, 
                     n_split:Optional[int]=5, topk:Optional[int]=5, metric:Optional[str]='P'):
    
    evals = pointwise_eval(pred_lbl, data_lbl, data_lbl_filterer, topk=topk, metric=metric)
    splits, info = equal_volume_split(data_lbl, n_split)
    values = [evals[:, split].sum()*100 for split in splits]
    return info, values


# %% ../nbs/16_analysis.ipynb 12
def barplot(scores:Dict, title:Optional[str]='', ylabel:Optional[str]='', figsize:Optional[Tuple]=(15,10), fname:Optional[str]=None):
    n_proc,n_split = len(scores),len(list(scores.values())[0][0])
    idx, width = np.arange(n_split), 0.8/n_proc
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid()

    shift = 0
    for proc,(info,values) in scores.items(): 
        x = idx + shift
        ax.bar(x=x, height=values, width=width, alpha=1, label=proc)
        shift += width

    shift = (n_proc//2)*width if n_proc%2 else width/2 + ((n_proc-1)//2)*width

    ax.set_title(title, fontsize=22)

    ax.set_xlabel('Quantiles \n (Increasing Freq.)', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    ax.set_xticks(idx + shift, info, fontsize=14)
    for o in ax.get_yticklabels(): o.set_fontsize(14)

    ax.legend(fontsize=14)
    
    if fname is not None: plt.savefig(fname)
    

# %% ../nbs/16_analysis.ipynb 13
def decile_plot(preds:Dict, data_lbl:sparse.csr_matrix, data_lbl_filterer:Optional[np.array]=None, 
                n_split:Optional[int]=5, topk:Optional[int]=5, metric:Optional[str]='P', figsize:Optional[Tuple]=(15,10), 
                title:Optional[str]='', fname:Optional[str]=None):
    scores = {}

    for method, pred in preds.items():
        info, values = get_decile_stats(pred, data_lbl, data_lbl_filterer, n_split=5, topk=5, metric='P')
        scores[method] = (info,values)
    
    barplot(scores, title, f'{metric}@{topk}', figsize, fname)
    

# %% ../nbs/16_analysis.ipynb 21
@typedispatch
def get_pred_dset(pred:sparse.csr_matrix, block:XCDataBlock):
    data = MainXCDataset(block.test.dset.data.data_info, pred, block.test.dset.data.lbl_info, 
                         block.test.dset.data.data_lbl_filterer)
    return XCDataset(data, **block.test.dset.meta)


@typedispatch
def get_pred_dset(pred:sparse.csr_matrix, dset:XCDataset):
    data = MainXCDataset(dset.data.data_info, pred, dset.data.lbl_info, dset.data.data_lbl_filterer)
    return XCDataset(data, **dset.meta)

@typedispatch
def get_pred_dset(pred:sparse.csr_matrix, dset:MainXCDataset):
    return MainXCDataset(dset.data_info, pred, dset.lbl_info, dset.data_lbl_filterer)
    

# %% ../nbs/16_analysis.ipynb 22
@typedispatch
def get_pred_sparse(out:XCPredictionOutput, n_lbl:int):
    pred_ptr = torch.concat([torch.zeros((1,), dtype=torch.long), out.pred_ptr.cumsum(dim=0)])
    return sparse.csr_matrix((out.pred_score, out.pred_idx, pred_ptr), shape=(len(out.pred_ptr), n_lbl))

@typedispatch
def get_pred_sparse(fname:str, n_lbl:int):
    with open(fname, 'rb') as f: out = pickle.load(f)
    pred_ptr = torch.concat([torch.zeros((1,), dtype=torch.long), out.pred_ptr.cumsum(dim=0)])
    return sparse.csr_matrix((out.pred_score, out.pred_idx, pred_ptr), shape=(len(out.pred_ptr), n_lbl))

@typedispatch
def load_pred_sparse(fname:str):
    o = np.load(fname)
    return sparse.csr_matrix((o['data'], o['indices'], o['indptr']), dtype=float, shape=o['shape'])

def get_output(fname:str, n_lbl:int, pred_type:Optional[str]='repr_output'):
    with open(pname, 'rb') as f: out = pickle.load(f)
    preds,targ = get_output_sparse(**getattr(out,pred_type), n_lbl=n_lbl)
    return preds, targ
    

# %% ../nbs/16_analysis.ipynb 23
def load_prediction_and_block(pkl_file:str, config_file:str, config_key:str, pred_file:str, use_sxc_sampler=True):
    block = build_block(pkl_file, config_file, use_sxc_sampler, config_key)
    pred = sparse.load_npz(pred_file)
    return pred, block
    

# %% ../nbs/16_analysis.ipynb 24
class PredictionBlock:

    def __init__(self, dset, pred, num_preds=10, pattern=r'.*_text'):
        self.dset, self.pred = dset, pred
        self.num_preds, self.pattern = num_preds, pattern
        if isinstance( self.dset, SXCDataset): self.dset = XCDataset._initialize(self.dset)

    @property
    def n_data(self): return self.dset.n_data
        
    @property
    def n_lbl(self): return self.dset.n_lbl

    def __getitem__(self, idx):
        prefix = 'pred2data'
        x = self.dset[idx]
        sort_idx = np.argsort(self.pred[idx].data)[:-self.num_preds-1:-1]
        x[f'{prefix}_idx'] = self.pred[idx].indices[sort_idx]
        if self.dset.data.lbl_info is not None:
            x.update({f'{prefix}_{k}':[v[i] for i in x[f'{prefix}_idx']] for k,v in self.dset.data.lbl_info.items() if k in self.dset.data.lbl_info_keys})
        return {k:v for k,v in x.items() if re.match(self.pattern, k)}
        

# %% ../nbs/16_analysis.ipynb 26
class TextDataset:
    
    def __init__(self, dset, pattern='.*_text$'):
        self.dset, self.pattern = dset, pattern
        colors = list(COLORS.keys())
        self.colors = [colors[i] for i in np.random.permutation(len(colors))]
    
    def __getitem__(self, idx):
        o = self.dset[idx]
        return {k:v for k,v in o.items() if re.match(self.pattern, k)}

    @property
    def n_data(self): return self.dset.n_data
        
    @property
    def n_lbl(self): return self.dset.n_lbl

    def show(self, idxs=None, k=10):
        if idxs is None: idxs = np.random.permutation(self.dset.n_data)[:k]
        for idx in idxs:
            for i,(k,v) in enumerate(self[idx].items()):
                key = colored(k, self.colors[i % len(self.colors)], attrs=["reverse", "blink"])
                value = colored(f': {v}', self.colors[i % len(self.colors)])
                print(key, value)
            print()

    def dump_txt(self, fname, idxs):
        with open(fname, 'w') as file:
            for idx in idxs:
                for i,(k,v) in enumerate(self[idx].items()):
                    file.write(f'{k}: {v}\n')
                file.write('\n')
            
    def dump_csv(self, fname, idxs):
        df = pd.DataFrame([self[idx] for idx in idxs])
        df.to_csv(fname, index=False)

    def dump(self, fname, idxs):
        if fname.endswith('.txt'): 
            self.dump_txt(fname, idxs)
        elif fname.endswith('.csv'): 
            self.dump_csv(fname, idxs)
        else: 
            raise ValueError(f'Invalid file extension: {fname}')
            

# %% ../nbs/16_analysis.ipynb 27
class CompareDataset:

    def __init__(self, dset1, dset2, dset1_prefix='1.', dset2_prefix='2.', pattern=r'.*_text'):
        self.dset1, self.dset2 = dset1, dset2
        self.dset1_prefix, self.dset2_prefix, self.pattern = dset1_prefix, dset2_prefix, pattern
        
        colors = list(COLORS.keys())
        self.colors = [colors[i] for i in np.random.permutation(len(colors))]

    def show(self, idxs=None, k=10):
        assert self.dset1.n_data == self.dset2.n_data, f'Different number of datapoints, dset1({self.dset1.n_data}) and dset2({self.dset2.n_data})'
        if idxs is None: idxs = np.random.permutation(self.dset1.n_data)[:k]

        for idx in idxs:
            items1, items2 = self.dset1[idx], self.dset2[idx]
            ctr = 0
            for k in items1.keys():
                if re.match(self.pattern, k):
                    color = self.colors[ctr % len(self.colors)]
                    
                    key = colored(f"{self.dset1_prefix} {k}", color, attrs=["reverse", "blink"])
                    value = colored(f': {items1[k]}', color)
                    print(key, value)
        
                    key = colored(f"{self.dset2_prefix} {k}", color, attrs=["reverse", "blink"])
                    value = colored(f': {items2[k]}', color)
                    print(key, value)

                    ctr += 1
            print()

    def dump_txt(self, fname, k=10):
        assert self.dset1.n_data == self.dset2.n_data, f'Different number of datapoints, dset1({self.dset1.n_data}) and dset2({self.dset2.n_data})'
        if idxs is None: idxs = np.random.permutation(self.dset1.n_data)[:k]
            
        with open(fname, 'w') as file:
            for idx in idxs:
                items1, items2 = self.dset1[idx], self.dset2[idx]
                for k in items1.keys():
                    if re.match(self.pattern, k):
                        file.write(f"{self.dset1_prefix} {k}: {items1[k]}\n")
                        file.write(f"{self.dset2_prefix} {k}: {items2[k]}\n")
                file.write('\n')
            

# %% ../nbs/16_analysis.ipynb 28
class Indices:

    def __init__(self, topk=10):
        self.topk = topk

    @staticmethod
    def head_data(dset, topk=10):
        return np.argsort(dset.data.data_lbl.getnnz(axis=1))[:-topk:-1]

    @staticmethod
    def tail_data(dset, topk=10):
        num = dset.data.data_lbl.getnnz(axis=1)
        idx = np.argsort(num)
        valid = (num > 0)[idx]
        return idx[valid][:topk]

    @staticmethod
    def metric(dset1, dset2, topk=10):
        prec_1 = pointwise_eval(dset1.pred, dset1.dset.data.data_lbl, dset1.dset.data.data_lbl_filterer)
        prec_2 = pointwise_eval(dset2.pred, dset2.dset.data.data_lbl, dset2.dset.data.data_lbl_filterer)
        
        scores = np.array(prec_1.sum(axis=1) - prec_2.sum(axis=1)).reshape(-1)
        idxs = np.argsort(scores)[:-topk-1:-1]
        return idxs
        
