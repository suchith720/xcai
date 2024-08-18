import torch
import pad_tfm

pad_transform = pad_tfm.PadTfm(pad_tok=0, pad_side='right', ret_t=True, in_place=True)
data = [[1, 2, 3], [4, 5]]
result = pad_transform(data, 0, "right", True, True)

