import os
from collections import OrderedDict

import torch

ckpt_file = 'work_dir/multitask/pretrained_run1/model_best_auc.pth'
header_key = 'backbone.'
target_folder = 'pretrained'
os.makedirs(target_folder, exist_ok=True)

output_file = os.path.join(target_folder, 'pretrained_run1_transferred.pth')
ckpt = torch.load(ckpt_file, map_location='cpu')

weights = OrderedDict([(k[len(header_key):], v)
                       for k, v in ckpt['model'].items()
                       if header_key in k])

new_ckpt = dict(
    model=weights,
    strict_load=False,
    input_size=ckpt['args_dict']['model_input_size'],
    input_nc=ckpt['args_dict']['channel'],
    normalize_mean=ckpt['norm_mean'],
    normalize_std=ckpt['norm_std'],
    arch=ckpt['args_dict']['backbone'],
)
torch.save(new_ckpt, output_file)
print(output_file)
