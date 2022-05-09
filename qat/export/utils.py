import os.path as osp

import torch.nn as nn


def replace_module_by_name(module: nn.Module, name: str, replace: nn.Module):
    if '.' in name:
        name, last_name = osp.splitext(name)
        last_name = last_name[1:]
        module = fetch_module_by_name(module, name)
    else:
        last_name = name
    try:
        idx = int(last_name)
        module[idx] = replace
    except:
        setattr(module, last_name, replace)


def fetch_module_by_name(module: nn.Module, name: str):
    for name in name.split('.'):
        try:
            idx = int(name)
            module = module[idx]
        except:
            module = getattr(module, name)
    return module
