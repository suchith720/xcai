# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/08_generation.trie.ipynb.

# %% auto 0
__all__ = ['TrieNode', 'TrieOutput', 'Trie', 'XCTrie']

# %% ../../nbs/08_generation.trie.ipynb 2
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional, List, Any, Union

from ..data import XCDataBlock
from fastcore.dispatch import *

# %% ../../nbs/08_generation.trie.ipynb 8
class TrieNode:
    def __init__(self, tok:int):
        self.tok, self.nxt_toks = tok, {}
        self.is_end, self.cnt, self.info = False, 0, None

    @property
    def data(self):
        return self.tok, self.nxt_toks, self.is_end, self.cnt, self.info

    @data.setter
    def data(self, x):
        self.tok, self.nxt_toks, self.is_end, self.cnt, self.info = x


# %% ../../nbs/08_generation.trie.ipynb 9
@dataclass
class TrieOutput:
    s:Optional[List]=None
    cnt:Optional[int]=None
    info:Optional[Any]=None
    

# %% ../../nbs/08_generation.trie.ipynb 10
class Trie(object):
    def __init__(self, max_info:Optional[int]=None):
        self.root, self.depth, self.max_info = None, 0, max_info

    @staticmethod
    def _add_info(node:TrieNode, info:Any):
        if node.info is None: 
            node.info = set(info) if isinstance(info, list) else set([info])
        else: 
            if isinstance(info, list): node.info.update(info)
            else: node.info.add(info)
        
    def insert(self, toks:Optional[List], info:Optional[Any]=None):
        if len(toks) > self.depth: self.depth = len(toks)
        if self.root is None: self.root=TrieNode(toks[0])
        if self.root.tok != toks[0]: raise ValueError(f'Expected `bos_tok` to be `{self.root.tok}` but got `{toks[0]}`.')
        node = self.root
        for tok in toks[1:]:
            node.cnt += 1
            if tok in node.nxt_toks: node = node.nxt_toks[tok]
            else: node.nxt_toks[tok]=node=TrieNode(tok)
        node.is_end = True
        if info is not None: Trie._add_info(node, info)
        node.cnt += 1
        
    @staticmethod
    def _search(node:TrieNode, p:List, o:List, max_info:Optional[int]=None):
        if node.is_end:
            info = list(node.info) if max_info is None else list(node.info)[:max_info]
            o.append(TrieOutput(p, node.cnt, info)); return
        for tok, n in node.nxt_toks.items(): Trie._search(n, p+[tok], o, max_info)

    def suffixes(self, x:Union[int,List]):
        x = [x] if isinstance(x, int) else x
        node, o = self.root, []
        if node.tok != x[0]: return []
        for tok in x[1:]:
            if tok in node.nxt_toks: node = node.nxt_toks[tok]
            else: return
        Trie._search(node, x, o, self.max_info)
        return sorted(o, key=lambda x: x.cnt, reverse=True)

    @staticmethod
    def _prune(node):
        for t,n in node.nxt_toks.items():
            Trie._prune(n)
            if len(node.nxt_toks) == 1 and len(n.nxt_toks) == 1 and next(iter(n.nxt_toks.values())).is_end:
                node.nxt_toks = n.nxt_toks
        
    def prune(self):
        self._prune(self.root)

    def prefix(self, x:List):
        node, o = self.root, [x[0]]
        if node.tok != x[0]: raise ValueError(f'`bos_tok`({x[0]}) cannot be "{node.tok}".')
        for tok in x[1:-1]:
            if tok in node.nxt_toks: node=node.nxt_toks[tok]; o.append(tok)
            else: break
        if x[-1] in node.nxt_toks and node.nxt_toks[x[-1]].is_end: return o+x[-1:]

    def __contains__(self, x:List):
        node = self.root
        if node.tok != x[0]: raise ValueError(f'`bos_tok`({x[0]}) cannot be "{node.tok}".')
        for tok in x[1:]: 
            if tok in node.nxt_toks: node = node.nxt_toks[tok]
            else: return False
        return node.is_end

    @property
    def bos_tok(self):
        return self.root.tok

    @typedispatch
    def update(self, x:List):
        for o in tqdm(x): self.insert(o)

    @typedispatch
    def update(self, x:List, y:List):
        for p,q in tqdm(zip(x,y), total=len(x)): self.insert(p,q)

    @classmethod
    @typedispatch
    def from_list(cls, x:List, max_info:Optional[int]=None):
        self = cls(max_info)
        for o in tqdm(x): self.insert(o)
        return self

    @classmethod
    @typedispatch
    def from_list(cls, x:List, y:List, max_info:Optional[int]=None):
        self = cls(max_info)
        for p,q in tqdm(zip(x,y), total=len(x)): self.insert(p,q)
        return self


# %% ../../nbs/08_generation.trie.ipynb 24
class XCTrie:
    
    @classmethod
    def from_block(cls, block:XCDataBlock, meta:Optional[List]=None, max_info:Optional[int]=None,
                   min_n_lbl:Optional[int]=1, max_n_lbl:Optional[int]=100):
        lbl_toks = block.lbl_info['input_ids']
        lbl_info = [[i] for i in range(len(lbl_toks))]
        
        trie = Trie.from_list(lbl_toks, lbl_info, max_info)

        if meta is not None:
            meta_dset = block.train.dset.meta
            for o in meta:
                if f'{o}_meta' not in meta_dset: raise ValueError(f'`{o}_meta` does not exist.')
                meta_lbl = meta_dset[f'{o}_meta'].lbl_meta.T.tocsr()
                n_lbl = meta_lbl.getnnz(axis=1)
                valid_meta_idx = np.where(np.logical_and(n_lbl>min_n_lbl, n_lbl<max_n_lbl))[0]
                
                meta_toks = [meta_dset[f'{o}_meta'].meta_info['input_ids'][i] for i in valid_meta_idx]
                meta_info = [o.indices.tolist() for o in tqdm(meta_lbl[valid_meta_idx], total=len(valid_meta_idx))]
                
                if len(meta_toks) != len(meta_info): raise ValueError(f'`meta_toks` and `meta_info` should have equal length.')
                trie.update(meta_toks, meta_info)
        return trie
        
