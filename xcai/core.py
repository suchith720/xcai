# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['show_data', 'Info', 'Filterer', 'get_tok_sparse', 'compute_inv_doc_freq', 'get_tok_idf', 'prepare_batch',
           'store_attr', 'get_attr', 'sorted_metric', 'display_metric', 'get_tensor_statistics', 'total_recall',
           'get_best_model', 'get_output_sparse', 'get_output', 'load_config', 'ScoreFusion', 'retain_randk',
           'random_topk', 'robustness_analysis', 'ShowMetric']

# %% ../nbs/00_core.ipynb 2
import pandas as pd, numpy as np, logging, sys, re, os, torch, json, inspect, torch.nn.functional as F

from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
from scipy import sparse
from tqdm.auto import tqdm
from itertools import chain
from scipy import sparse
from IPython.display import display
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from typing import List, Dict, Union, Optional, Any, Callable
from torch.utils.data import Dataset

from fastcore.dispatch import *
from fastcore.basics import *

# %% ../nbs/00_core.ipynb 5
def show_data(x:Dict, n:Optional[int]=10, seed:Optional[int]=None):
    with pd.option_context('display.max_colwidth', None):
        display(pd.DataFrame(x).sample(n, random_state=seed))
        

# %% ../nbs/00_core.ipynb 6
class Info():

    def __init__(self):
        self.tokz, self.info = None, None
        
    @staticmethod
    def _read_txt(fname:str, sep:Optional[str]='->', enc:Optional[str]='latin-1'):
        with open(fname, encoding=enc) as f:
            info = [o[:-1] for o in f]
        return list(zip(*[(o,) if sep is None else o.split(sep, maxsplit=1) for o in info]))

    @staticmethod
    def _read_csv(fname:str):
        df = pd.read_csv(fname).fillna('')
        return [df[c].tolist() for c in df.columns]

    @staticmethod
    def _read_file(fname:str, sep:Optional[str]='->', enc:Optional[str]='latin-1'):
        if fname.endswith(".txt"): 
            return Info._read_txt(fname, sep=sep, enc=enc)
        elif fname.endswith(".csv"): 
            return Info._read_csv(fname)
        else: 
            raise ValueError(f"Invalid filename: {fname}.")
        
    @staticmethod
    def _read_info(fname:str, sep:Optional[str]='->', info_column_names:Optional[List]=None, enc:Optional[str]='latin-1'):
        info = Info._read_file(fname, sep=sep, enc=enc)
        info_column_names = list(range(len(info))) if info_column_names is None else info_column_names
        if len(info_column_names) != len(info): raise ValueError(f'`info_column_names` and `info` should have same number of elements.')
        return {p:q for p,q in zip(info_column_names, info)}

    def read_info(self, fname:Optional[str], sep:Optional[str]='->', info_column_names:Optional[List]=None, enc:Optional[str]='latin-1'):
        self.info = Info._read_info(fname, sep, info_column_names, enc)
        return self.info
    
    def tokenize(self, tokenization_column:Union[int, str], tokenizer:Union[str, PreTrainedTokenizerBase], 
                 max_sequence_length:Optional[int]=None, padding:Optional[bool]=True, return_tensors:Optional[str]=None, 
                 prompt_func:Optional[Callable]=None):
        if self.tokz is None: self.tokz = tokenizer if isinstance(tokenizer, PreTrainedTokenizerBase) else AutoTokenizer.from_pretrained(tokenizer)
        tokenization_column = list(self.info.keys())[0] if tokenization_column is None else tokenization_column
        if tokenization_column is None: logging.info(f'`tokenization_column` not given as input, so value set to {tokenization_column}.')
        if tokenization_column not in self.info: raise ValueError(f'`{tokenization_column}` is invalid `tokenization_column` value.')

        tokenization_text = self.info[tokenization_column] if prompt_func is None else [prompt_func(t) for t in self.info[tokenization_column]]
        self.info.update(self.tokz(tokenization_text, truncation=True, max_length=max_sequence_length, 
                                   padding=padding, return_tensors=return_tensors))
        
        return self.info

    def show_data(self, n:Optional[int]=10, seed:Optional[int]=None):
        with pd.option_context('display.max_colwidth', None):
            display(pd.DataFrame(self.info).sample(n, random_state=seed))

    def __len__(self):
        if self.info is None: return 0
        n_info = [len(v) for v in self.info.values()]
        if len(n_info) == 0: raise ValueError('`info` cannot be empty.')
        if not np.all([o == n_info[0] for o in n_info]): raise ValueError('`info` should contain features with same length.')
        return n_info[0]

    @classmethod
    def from_txt(cls, 
                 fname:str, 
                 sep:Optional[str]='->', 
                 info_column_names:Optional[List]=None, 
                 enc:Optional[str]='latin-1',
                 use_tokenizer:Optional[bool]=False,
                 tokenizer:Optional[Union[str,PreTrainedTokenizerBase]]=None,
                 tokenization_column:Optional[str]=None,
                 max_sequence_length:Optional[int]=None,
                 padding:Optional[bool]=True,
                 return_tensors:Optional[str]=None,
                 prompt_func:Optional[Callable]=None,
                 **kwargs):
        self = cls()
        self.info = self.read_info(fname, sep, info_column_names, enc)
        if use_tokenizer: 
            self.tokenize(tokenization_column, tokenizer, max_sequence_length, padding=padding, return_tensors=return_tensors, 
                          prompt_func=prompt_func)
        return self.info
        

# %% ../nbs/00_core.ipynb 22
class Filterer:

    @staticmethod
    def load_filter(fname:str):
        if fname is not None and os.path.exists(fname): return np.loadtxt(fname, dtype=np.int64)
        
    @staticmethod
    def generate(train_id:List, test_id:List, lbl_id:List, train_lbl:sparse.csr_matrix, test_lbl:sparse.csr_matrix):
        _, train_idx, lbl2train_idx = np.intersect1d(train_id, lbl_id, return_indices=True)
        train_lbl_filterer = np.vstack([train_idx, lbl2train_idx]).T
        
        _, test_idx, lbl2test_idx = np.intersect1d(test_id, lbl_id, return_indices=True)
        test_lbl_filterer = np.vstack([test_idx, lbl2test_idx]).T
        
        train_udx, train_udx2idx = np.unique(train_idx, return_index=True)
        lbl2test_udx, lbl2test_udx2idx = np.unique(lbl2test_idx, return_index=True)
        
        _test_lbl_filterer = train_lbl[train_udx][:, lbl2test_udx].T
        
        rows, cols = _test_lbl_filterer.nonzero()
        test_idx = test_idx[lbl2test_udx2idx[rows]]
        lbl2test_idx = lbl2train_idx[train_udx2idx[cols]]
        
        _test_lbl_filterer = np.vstack([test_idx, lbl2test_idx]).T
        test_lbl_filterer = np.vstack([test_lbl_filterer, _test_lbl_filterer])
    
        return train_lbl_filterer, test_lbl_filterer

    @staticmethod
    def sample(f:np.array, sz:tuple, idx:List):
        f = sparse.coo_matrix((np.full(f.shape[0],1), (f[:, 0], f[:, 1])), shape=sz).tocsr()
        f = f[idx].tocoo()
        return np.vstack([f.row, f.col]).T

    @staticmethod
    def prune(data:sparse.csr_matrix, data_filterer:np.array):
        data = data.copy()
        data[data_filterer[:,0], data_filterer[:,1]] = 0
        data.eliminate_zeros()
        
        idx = np.where(data.getnnz(axis=1) > 0)[0]
        return data[idx], Filterer.sample(data_filterer, data.shape, idx), idx

    @staticmethod
    def apply(data:sparse.csr_matrix, data_filterer:np.array):
        data[data_filterer[:,0], data_filterer[:,1]] = 0
        data.eliminate_zeros()
        return data

        

# %% ../nbs/00_core.ipynb 24
def get_tok_sparse(tokens:List, n_cols:Optional[int]=None):
    n_toks = torch.tensor([len(tok) for tok in tokens])
    tok_ptr = torch.concat([torch.zeros((1,), dtype=torch.long), n_toks.cumsum(dim=0)])
    toks = torch.tensor(list(chain(*tokens)))
    tok_cnt = torch.full((toks.shape[0],), 1, dtype=torch.long)
    m = sparse.csr_matrix((tok_cnt, toks, tok_ptr)) if n_cols is None else sparse.csr_matrix((tok_cnt, toks, tok_ptr), shape=(len(n_toks), n_cols))
    m.sum_duplicates()
    return m

def compute_inv_doc_freq(inputs:sparse.csr_matrix):
    n_docs = inputs.shape[0]
    doc_freq = torch.tensor(inputs.getnnz(axis=0))
    return torch.log((n_docs+1)/(doc_freq+1))+1

def get_tok_idf(dset:Dataset, field:Optional[str]='data_input_ids', n_cols:Optional[int]=None):
    toks = [list(chain(*dset[i][field])) for i in tqdm(range(len(dset)))]
    tok_sparse = get_tok_sparse(toks, n_cols=n_cols)
    return compute_inv_doc_freq(tok_sparse)
    

# %% ../nbs/00_core.ipynb 26
def prepare_batch(m, b, m_args=None):
    m_kwargs = inspect.signature(m.forward).parameters
    return BatchEncoding({k:v for k,v in b.items() if k in m_kwargs or (m_args is not None and k in m_args)})
    

# %% ../nbs/00_core.ipynb 27
def store_attr(names=None, self=None, but='', cast=False, store_args=None, is_none=True, **attrs):
    fr = sys._getframe(1)
    args = argnames(fr, True)
    if self: args = ('self', *args)
    else: self = fr.f_locals[args[0]]
    if store_args is None: store_args = not hasattr(self,'__slots__')
    if store_args and not hasattr(self, '__stored_args__'): self.__stored_args__ = {}
    anno = annotations(self) if cast else {}
    if names and isinstance(names,str): names = re.split(', *', names)
    ns = names if names is not None else getattr(self, '__slots__', args[1:])
    added = {n:fr.f_locals[n] for n in ns}
    attrs = {**attrs, **added}
    if isinstance(but,str): but = re.split(', *', but)
    attrs = {k:v for k,v in attrs.items() if k not in but}
    return _store_attr(self, anno, is_none, **attrs)
    

# %% ../nbs/00_core.ipynb 28
def _store_attr(self, anno, is_none, **attrs):
    stored = getattr(self, '__stored_args__', None)
    for n,v in attrs.items():
        if n in anno: v = anno[n](v)
        if is_none or v is not None: setattr(self, n, v)
        if stored is not None: stored[n] = v
       

# %% ../nbs/00_core.ipynb 29
def get_attr(x, attr:str):
    for a in attr.split('.'): x = getattr(x, a)
    return x

# %% ../nbs/00_core.ipynb 30
def sorted_metric(keys:List, order:Optional[Dict]=None):
    order = {o.split('@')[0]:i for i,o in enumerate(keys)}
    def _suffix(x): return (int(x.split('_')[0]) , x.split('_')[1]) if '_' in x else (int(x),)
    def _key_fn(x): return order[x[0]], _suffix(x[1])
    def key_fn(x): return _key_fn(x.split('@'))
    return sorted(keys, key=key_fn)

def display_metric(metrics, remove_prefix:Optional[bool]=True, order:Optional[List]=None, scale:Optional[int]=100.0):
    metrics = {k.split('_', maxsplit=1)[1]:v for k,v in metrics.items()} if remove_prefix else metrics
    metric_keys, other_keys = sorted_metric([k for k in metrics if '@' in k], order), [k for k in metrics if '@' not in k]
    
    from IPython.display import display
    with pd.option_context('display.precision',4,'display.max_colwidth',None,'display.max_columns',None):
        metric,other = pd.DataFrame([metrics])[metric_keys]*scale, pd.DataFrame([metrics])[other_keys]
        display(pd.concat([metric, other], axis=1))
        

# %% ../nbs/00_core.ipynb 31
def get_tensor_statistics(x:torch.Tensor):
    c = ['mean', 'std', '25', '50', '75']
    s = torch.cat([x.float().mean(dim=0, keepdim=True), 
                   x.float().std(dim=0, keepdim=True),
                   torch.quantile(x.float(), torch.tensor([0.25, 0.5, 0.75]))])
    return pd.DataFrame([s.tolist()], columns=c)

def total_recall(inp_idx:torch.Tensor, n_inp:torch.Tensor, targ:sparse.csr_matrix, filterer:sparse.csr_matrix):
    val, ptr = torch.ones(len(inp_idx)), torch.cat([torch.zeros(1, dtype=torch.int64), n_inp.cumsum(0)])
    inp = sparse.csr_matrix((val,inp_idx,ptr), shape=targ.shape); inp.sum_duplicates(); inp.data[:] = 1
    if filterer is not None: inp, targ = Filterer.apply(inp, filterer), Filterer.apply(targ, filterer)
    sc = inp.multiply(targ)/(targ.getnnz(axis=1)[:, None]*targ.shape[0])
    return sc.sum(), sc
    

# %% ../nbs/00_core.ipynb 32
def get_best_model(mdir:str, pat:Optional[str]=r'^checkpoint-(\d+)'):
    nm = sorted([int(re.match(pat, o).group(1)) for o in os.listdir(mdir) if re.match(pat, o)])[-1]
    fname = f'{mdir}/checkpoint-{nm}/trainer_state.json'
    with open(fname, 'r') as file: mname = json.load(file)['best_model_checkpoint']
    return f'{mdir}/checkpoint-{nm}' if mname is None else mname
    

# %% ../nbs/00_core.ipynb 33
def get_output_sparse(pred_idx, pred_ptr, pred_score, targ_idx, targ_ptr, n_lbl):
    n_data = pred_ptr.shape[0]
    
    pred_ptr = torch.cat([torch.zeros((1,), dtype=torch.long), pred_ptr.cumsum(dim=0)])
    
    targ_ptr = torch.cat([torch.zeros((1,), dtype=torch.long), targ_ptr.cumsum(dim=0)])
    targ_score = torch.ones((targ_idx.shape[0],), dtype=torch.long)
    
    pred = sparse.csr_matrix((pred_score,pred_idx,pred_ptr), shape=(n_data, n_lbl))
    targ = sparse.csr_matrix((targ_score,targ_idx,targ_ptr), shape=(n_data, n_lbl))
    return pred, targ


# %% ../nbs/00_core.ipynb 34
def get_output(data_lbl, pred_lbl):
    output = {
        'targ_idx': torch.tensor(data_lbl.indices),
        'targ_ptr': torch.tensor([q-p for p,q in zip(data_lbl.indptr, data_lbl.indptr[1:])]),
        'pred_idx': torch.tensor(pred_lbl.indices),
        'pred_ptr': torch.tensor([q-p for p,q in zip(pred_lbl.indptr, pred_lbl.indptr[1:])]),
        'pred_score': torch.tensor(pred_lbl.data),
    }
    return output

# %% ../nbs/00_core.ipynb 35
def load_config(fname, key):
    with open(fname, 'r') as file:
        return json.load(file)[key]
        

# %% ../nbs/00_core.ipynb 37
class ScoreFusion():
    
    def __init__(self, prop:Optional[np.array]=None, max_depth:Optional[int]=7):
        self.clf, self.prop = DecisionTreeClassifier(max_depth=max_depth), prop
        
    def sample(self, 
               score_a:sparse.csr_matrix, 
               score_b:sparse.csr_matrix, 
               targ:Optional[sparse.csr_matrix]=None, 
               n_samples:Optional[int]=None):
        if n_samples is not None and n_samples > 0 and n_samples < score_a.shape[0]:
            rnd_idx = np.random.permutation(score_a.shape[0])[:n_samples]
            score_a, score_b = score_a[rnd_idx], score_b[rnd_idx]
            targ = targ if targ is None else targ[rnd_idx]
        return score_a, score_b, targ
    
    def get_inp(self, row_idx, col_idx, score_a, score_b):
        inp = [np.array(o[row_idx,col_idx]).reshape(-1, 1) for o in [score_a, score_b]]
        if self.prop is not None: inp.append(self.prop[col_idx].reshape(-1,1))
        return np.hstack(inp)
    
    def prepare_inputs(self, score_a:sparse.csr_matrix, score_b:sparse.csr_matrix, 
                       targ:Optional[sparse.csr_matrix]=None, n_samples:Optional[int]=None):
        if n_samples is not None:
            score_a, score_b, targ = self.sample(score_a, score_b, targ, n_samples)
        row_idx, col_idx = score_b.nonzero()
        inp = self.get_inp(row_idx, col_idx, score_a, score_b)
        targ = targ if targ is None else np.array(targ[row_idx, col_idx]).ravel()
        return inp, targ, (row_idx,col_idx)
        
    def fit(self, score_a:sparse.csr_matrix, score_b:sparse.csr_matrix, targ:Optional[sparse.csr_matrix]=None, 
            n_samples:Optional[int]=None):
        score, targ, _ = self.prepare_inputs(score_a, score_b, targ, n_samples)
        self.clf.fit(score, targ)

    def predict(self, score_a:sparse.csr_matrix, score_b:sparse.csr_matrix, beta:Optional[int]=1):
        inp, _, (row_idx, col_idx) = self.prepare_inputs(score_a, score_b)
        res = score_b.copy()
        res[row_idx,col_idx] = self.clf.predict_proba(inp)[:, 1]
        return beta*(res+score_a)+score_b
    

# %% ../nbs/00_core.ipynb 39
def retain_randk(matrix:sparse.csr_matrix, topk:Optional[int]=3):
    data, indices, indptr = [], [], np.zeros_like(matrix.indptr)
    for i,row in tqdm(enumerate(matrix), total=matrix.shape[0]):
        if row.nnz > 0:
            idx = np.random.randint(row.nnz, size=topk)
            ind, d = row.indices[idx], row.data[idx]
            indptr[i+1] = indptr[i] + topk
            indices.append(ind); data.append(d)
        else:
            indptr[i+1] = indptr[i]
    
    data = np.hstack(data)
    indices = np.hstack(indices)
    
    o = sparse.csr_matrix((data, indices, indptr), dtype=matrix.dtype)
    o.sort_indices()
    return o
    

# %% ../nbs/00_core.ipynb 41
def random_topk(data_lbl, topk=5):
    data,indices,indptr = [],[],[0]
    for i,j in tqdm(zip(data_lbl.indptr, data_lbl.indptr[1:]), total=data_lbl.shape[0]):
        idx = np.random.permutation(j-i)[:topk]
        data.append(data_lbl.data[i:j][idx])
        indices.append(data_lbl.indices[i:j][idx])
        indptr.append(indptr[-1]+len(idx))
    data = np.hstack(data)
    indices = np.hstack(indices)
    indptr = np.array(indptr)

    o = sparse.csr_matrix((data,indices,indptr), shape=data_lbl.shape, dtype=np.float32)
    o.sort_indices()
    o.eliminate_zeros()
    return o
    

# %% ../nbs/00_core.ipynb 42
def robustness_analysis(block, meta_name:str, analysis_type:str='missing', pct:float=0.5, topk:int=3):
    data_meta = random_topk(block.test.dset.meta[f'{meta_name}_meta'].data_meta, topk=topk)
    
    if analysis_type == 'noise':
        mask = np.random.rand(len(data_meta.data)) < pct
        data_meta.indices[mask] = np.random.randint(block.n_lbl, size=mask.sum())
        data_meta.sort_indices()
    elif analysis_type == 'missing':
        data_meta.data[np.random.rand(len(data_meta.data)) < pct] = 0
        data_meta.eliminate_zeros()
    else:
        raise ValueError(f'Invalid `analysis_type`: {analysis_type}.')
        
    lbl_meta = block.test.dset.meta[f'{meta_name}_meta'].lbl_meta
    block.test.dset.meta[f'{meta_name}_meta'].update_meta_matrix(data_meta, lbl_meta)
    return block
    

# %% ../nbs/00_core.ipynb 44
class ShowMetric:

    ORDER = ['P@1', 'P@5', 'N@5', 'PSP@1', 'PSP@5', 'R@200']

    @staticmethod
    def show_df(df):
        with pd.option_context('display.precision',2,'display.max_colwidth',None,'display.max_columns',None):
            display(df)

    @staticmethod
    def convert_df_and_remove_prefix(o, order=None):
        m = {}
        for key,val in o.items():
            m[key] = {re.sub(r'^(test|eval)_(.*)', r'\2', k):v for k,v in val.items()}
            
        df = pd.DataFrame(m).T
        order = [o for o in ShowMetric.ORDER if o in df.columns] if order is None else order
        if isinstance(df.index[0], tuple): df.index = pd.MultiIndex.from_tuples(df.index)
        return df[order]

    @staticmethod
    def show(o, order=None):
        ShowMetric.show_df(ShowMetric.convert_df_and_remove_prefix(o, order)*100)
        
