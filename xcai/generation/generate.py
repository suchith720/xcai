# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/09_generation.generate.ipynb.

# %% auto 0
__all__ = ['TriePtr', 'Hypothesis', 'TrieBeam', 'tbs_proc', 'TrieBeamSearch']

# %% ../../nbs/09_generation.generate.ipynb 3
import torch, math
from torch.multiprocessing import Pool
import torch.nn.functional as F
from itertools import chain
from tqdm.auto import tqdm
from typing import Optional, Sequence, Any, Dict, List
from dataclasses import dataclass

from fastcore.utils import *
from fastcore.dispatch import *
from fastcore.meta import *
from fastcore.parallel import *

from ..core import *
from ..transform import *
from .trie import *

# %% ../../nbs/09_generation.generate.ipynb 12
class TriePtr:

    def __init__(self, trie, max_info:Optional[int]=None):
        store_attr('trie,max_info')
        self.ptr, self.hyp = trie.root, [trie.root.tok]

    @property
    def tokens(self):
        return list(self.ptr.nxt_toks.keys())

    def next(self, val:int):
        if val not in self.tokens: raise ValueError(f'`{val}` not a valid next token.')
        self.ptr = self.ptr.nxt_toks[val]
        self.hyp.append(val)

    def suffixes(self):
        o = []
        Trie._search(self.ptr, self.hyp, o, self.max_info)
        return sorted(o, key=lambda x: x.cnt, reverse=True)

    @property
    def is_end(self):
        return self.ptr.is_end

    @property
    def value(self):
        info = list(self.ptr.info) if self.max_info is None else list(self.ptr.info)[:self.max_info]
        return TrieOutput(self.hyp, self.ptr.cnt, info)

    def copy(self):
        t = TriePtr(self.trie, self.max_info)
        t.ptr,t.hyp = self.ptr,self.hyp.copy()
        return t
        

# %% ../../nbs/09_generation.generate.ipynb 26
class Hypothesis:

    def __init__(self, n_bm:int, len_penalty:Optional[float]=1.0):
        store_attr('n_bm,len_penalty')
        self.worst_sc, self.beams = 1e9, []

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logits:float, gen_len:Optional[int]=None):
        if gen_len is not None: sc = sum_logits/gen_len**self.len_penalty
        else: sc = sum_logits/len(hyp.s)**self.len_penalty

        if len(self) < self.n_bm or sc > self.worst_sc:
            self.beams.append((sc, hyp))
            if len(self) > self.n_bm:
                nxt_sc = sorted([(s,i) for i,(s,_) in enumerate(self.beams)])
                del self.beams[nxt_sc[0][1]]
                self.worst_sc = nxt_sc[1][0]
            else: self.worst_sc = min(sc, self.worst_sc)

    def is_done(self, best_sc:float, cur_len:int):
        if len(self) < self.n_bm: return False
        high_sc = best_sc/cur_len**self.len_penalty
        return self.worst_sc >= high_sc
        

# %% ../../nbs/09_generation.generate.ipynb 33
class TrieBeam:

    def __init__(self, trie:Trie, n_bm:Optional[int]=5, max_bm:Optional[int]=None, len_penalty:Optional[float]=1.0, 
                 max_info:Optional[int]=None):
        store_attr('trie,n_bm,len_penalty,max_info')
        self.max_bm, self.hyp = max_bm if max_bm is None else max(max_bm, 2*n_bm), None

    def valid(self, ptr:List, sc:torch.FloatTensor):
        v_tok, v_sc, v_idx = [], [], []
        for i,(p,s) in enumerate(zip(ptr,sc)):
            toks = p.tokens
            v_tok.extend(toks)
            v_sc.extend(s[toks].tolist())
            v_idx.extend([i for _ in range(len(toks))])
        return v_tok, v_sc, v_idx

    def topk(self, ptr:List, tok:List, sc:List, idx:List):
        top_sc, top_i = (
            torch.topk(torch.tensor(sc), 2*self.n_bm, dim=0) 
            if len(sc) > 2*self.n_bm else torch.sort(torch.tensor(sc), dim=0, descending=True)
        )
        top_sc = top_sc.tolist()
        top_idx, top_tok = list(zip(*[(idx[i],tok[i]) for i in top_i]))
        top_ptr = [ptr[i].copy() for i in top_idx]
        for p,t in zip(top_ptr, top_tok): p.next(t)
        return top_ptr, top_sc

    def next(self, ptr:List, sc:List):
        nxt_ptr, nxt_sc = [], []
        for i,(p,s) in enumerate(zip(ptr, sc)):
            if p.is_end: self.hyp.add(p.value, s)
            else: nxt_ptr.append(p);nxt_sc.append(s)
        nxt_ptr,nxt_sc = nxt_ptr[:self.n_bm],torch.tensor(nxt_sc[:self.n_bm]).unsqueeze(1)
        return nxt_ptr, nxt_sc

    def finalize(self, ptr:List, sc:List):
        if len(self.hyp) < self.n_bm:
            nh = int(math.ceil((self.max_bm-len(self.hyp))/len(ptr))) if self.max_bm is not None and len(ptr) else None
            for p,s in zip(ptr, sc):
                hyps = p.suffixes() if nh is None else p.suffixes()[:nh]
                for o in hyps: self.hyp.add(o, s)
        if len(self.hyp) < self.n_bm: raise ValueError(f'`len(self.hyp)`({len(self.hyp)}) < `n_bm`({self.n_bm})')
        seq_sc, seq_ids, info, n_info = list(map(list, zip(*[(sc,hyp.s,hyp.info,len(hyp.info)) for sc,hyp in self.hyp.beams])))
        return {
            'seq2data_data2ptr':[self.n_bm],
            'seq2data_score':seq_sc, 
            'seq2data_output_ids':seq_ids, 
            'info2seq2data_idx':list(chain(*info)),
            'info2seq2data_seq2ptr':n_info,
            'info2seq2data_data2ptr':[sum(n_info)],
        }
        
    def proc(self, logits:torch.FloatTensor, n_bm:Optional[int]=None, max_bm:Optional[int]=None, len_penalty:Optional[float]=None, 
             max_info:Optional[int]=None):
        store_attr('n_bm,len_penalty,max_info', is_none=False)
        if max_bm is not None: self.max_bm = max(max_bm, 2*self.n_bm)
        
        self.hyp = Hypothesis(self.n_bm, self.len_penalty)
        sc = torch.full((self.n_bm,1), -1e9); sc[0,0] = 0
        ptr = [TriePtr(self.trie,self.max_info) for _ in range(2*self.n_bm)]
        
        cur_len,seq_len = 1,logits.shape[0]
        while True:
            sc = logits[cur_len:cur_len+1].expand(sc.shape[0],-1) + sc
            v_tok, v_sc, v_idx = self.valid(ptr, sc)
            top_ptr, top_sc = self.topk(ptr, v_tok, v_sc, v_idx)
            ptr, sc = self.next(top_ptr, top_sc)
            cur_len += 1
            
            if cur_len >= seq_len or len(ptr) == 0 or self.hyp.is_done(sc.max().item(), cur_len):
                break
        return self.finalize(ptr, sc.squeeze(1).tolist())
        

# %% ../../nbs/09_generation.generate.ipynb 40
def tbs_proc(x): return x[0].proc(x[1])

# %% ../../nbs/09_generation.generate.ipynb 41
class TrieBeamSearch:

    @delegates(XCPadOutputTfm.__init__)
    def __init__(self, trie:Trie, n_bm:int=5, max_bm:Optional[int]=None, len_penalty:Optional[float]=1.0, max_info:Optional[int]=None,
                 n_threads=3, **kwargs):
        store_attr('trie,n_bm,max_bm,len_penalty,max_info,n_threads')
        self.tfm = XCPadOutputTfm(**kwargs)
        
    def proc(self, model, inputs:Dict, n_bm:int=None, max_bm:Optional[int]=None, len_penalty:Optional[float]=None, 
             max_info:Optional[int]=None):
        store_attr('n_bm,max_bm,len_penalty,max_info', is_none=False)
        logits, attention_mask = F.log_softmax(model(**inputs).logits, dim=-1).cpu(), inputs['data_attention_mask'].bool().cpu()
        hyps = [TrieBeam(self.trie, self.n_bm, self.max_bm, self.len_penalty, self.max_info) for _ in range(logits.shape[0])]
        outputs = [h.proc(l[a]) for h,l,a in zip(hyps, logits, attention_mask)]
        outputs = self.tfm({k:list(chain(*[o[k] for o in outputs])) for k in outputs[0]})
        outputs['info2seq2data_score'] = torch.repeat_interleave(outputs['seq2data_score'], outputs['info2seq2data_seq2ptr'], dim=0)
        return outputs

    def proc_parallel(self, model, inputs:Dict, n_bm:int=None, max_bm:Optional[int]=None, len_penalty:Optional[float]=None, 
                      max_info:Optional[int]=None, n_threads=None):
        store_attr('n_bm,max_bm,len_penalty,max_info,n_threads', is_none=False)
        logits = F.log_softmax(model(**inputs).logits, dim=-1).cpu().share_memory_()
        attention_mask = inputs['data_attention_mask'].bool().cpu().share_memory_()
        hyps = [TrieBeam(self.trie, self.n_bm, self.max_bm, self.len_penalty, self.max_info) for _ in range(logits.shape[0])]
        
        with torch.no_grad(), Pool(processes=n_threads) as pool: outputs = list(pool.map(tbs_proc, list(zip(hyps, logits, attention_mask))))
        
        outputs = self.tfm({k:list(chain(*[o[k] for o in outputs])) for k in outputs[0]})
        outputs['info2seq2data_score'] = torch.repeat_interleave(outputs['seq2data_score'], outputs['info2seq2data_seq2ptr'], dim=0)
        return outputs
        
