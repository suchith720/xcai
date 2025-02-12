# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/36_main.ipynb.

# %% auto 0
__all__ = ['main_build_block', 'main_load_model', 'get_output', 'main_run']

# %% ../nbs/36_main.ipynb 1
import os, torch, scipy.sparse as sp, pickle
from .block import SXCBlock, XCBlock
from typing import Optional, Dict, Callable

# %% ../nbs/36_main.ipynb 3
def main_build_block(pkl_file, config, use_sxc:Optional[bool], config_key:Optional[str]=None, **kwargs):
        if os.path.exists(config): config = load_config(config, config_key)

        if use_sxc: block = SXCBlock.from_cfg(config, padding=True, return_tensors='pt', **kwargs)
        else: block = XCBlock.from_cfg(config, transform_type='xcs')
        with open(pkl_file, 'wb') as file: pickle.dump(block, file)


# %% ../nbs/36_main.ipynb 4
def main_load_model(output_dir:str, model_fn:Callable, model_args:Dict, init_fn:Callable,
        init_args:Optional[Dict]=dict(), do_inference:Optional[bool]=False):
    if do_inference:
        os.environ['WANDB_MODE'] = 'disabled'
        mname = os.path.basename(get_best_model(output_dir))
        model_args.mname = f'{args.output_dir}/{mname}'

    model = model_fn(**model_args)
    if do_inference: init_fn(model, **init_args)

    return model
    

# %% ../nbs/36_main.ipynb 5
def get_output(pred_idx:torch.Tensor, pred_ptr:torch.Tensor, pred_score:torch.Tensor, n_lbl:int, **kwargs):
    n_data = pred_ptr.shape[0]
    pred_ptr = torch.cat([torch.zeros((1,), dtype=torch.long), pred_ptr.cumsum(dim=0)])
    pred = sp.csr_matrix((pred_score,pred_idx,pred_ptr), shape=(n_data, n_lbl))
    return pred
    

# %% ../nbs/36_main.ipynb 6
def main_run(learn, parse_args, n_lbl:int, do_inference:bool):
    if do_prediction:
        pred_dir = f'{learn.args.output_dir}/predictions'
        os.makedirs(pred_dir, exist_ok=True)

        if parse_args.save_repr:
            trn_repr, lbl_repr = learn.get_data_and_lbl_representation(block.train.dset)
            tst_repr = learn._get_data_representation(block.test.dset)

            torch.save(trn_repr, f'{pred_dir}/train_repr.pth')
            torch.save(tst_repr, f'{pred_dir}/test_repr.pth')
            torch.save(lbl_repr, f'{pred_dir}/label_repr.pth')

        if parse_args.do_test_inference:
            o = learn.predict(block.test.dset)
            print(o.metrics)

            if parse_args.save_test_inference:
                with open(f'{pred_dir}/test_predictions.pkl', 'wb') as file:
                    pickle.dump(o, file)

                pred = get_output(o.pred_idx, o.pred_ptr, o.pred_score, n_lbl=n_lbl)
                sp.save_npz(f'{pred_dir}/test_predictions.npz', pred)

        if parse_args.do_train_inference:
            o = learn.predict(block.train.dset)
            print(o.metrics)

            if parse_args.save_train_inference:
                with open(f'{pred_dir}/train_predictions.pkl', 'wb') as file:
                    pickle.dump(o, file)

                pred = get_output(o.pred_idx, o.pred_ptr, o.pred_score, n_lbl=n_lbl)
                sp.save_npz(f'{pred_dir}/train_predictions.npz', pred)
    else:
        learn.train()
        
