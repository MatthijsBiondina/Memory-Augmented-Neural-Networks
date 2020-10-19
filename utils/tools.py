from typing import Any

import shutil
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import collections
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm


def nan(x):
    if int(torch.sum(torch.isnan(x)).detach().cpu().numpy()):
        return


def random_chance(p: float = 0.5) -> bool:
    return (np.random.random_sample() < p)


def pyout(*args, ex=None):
    """
    Print with part of trace for debugging.

    :param args: arguments to print
    :param ex: if not None, exits application with provided exit code
    :return:
    """
    trace = traceback.format_stack()[-2].split('\n')
    _tqdm_write("\033[1;33m" + trace[0].split(', ')[0].replace('  ', '') + "\033[0m")
    _tqdm_write("\033[1;33m" + trace[0].split(', ')[1].split(' ')[1] + ':', trace[1].replace('    ', ''), "\033[0m")
    _tqdm_write(*args)
    _tqdm_write("")
    if ex is not None:
        sys.exit(ex)


def poem(desc: Any) -> str:
    """
    Format description for tqdm bar. Assumes desired width of 23 characters for description. Adds whitespace if
    description is shorter, clips if description is longer.

    :param desc: description
    :return: formatted description
    """
    desc = str(desc)

    if len(desc) < 23:
        return desc + ' ' * (23 - len(desc))
    else:
        return desc[:20] + '...'


def _tqdm_write(*args):
    tqdm.write(datetime.now().strftime("%d-%m-%Y %H:%M"), end=' ')
    for arg in list(map(str, args)):
        for ii, string in enumerate(arg.split('\n')):
            tqdm.write((' ' * 17 if ii > 0 else '') + string, end=' ' if ii == len(arg.split('\n')) - 1 else '\n')
    tqdm.write('')


def listdir(path, end=None, contains=None):
    return sorted(os.path.join(path, file) for file in os.listdir(path)
                  if (1 if end is None else file.endswith(end)) and (1 if contains is None else contains in file))


def fname(path):
    return path.split('/')[-1]


def makedir(path, delete=False):
    if delete:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
    os.makedirs(path, exist_ok=True)


def watch(loc):
    ou = {}
    for name, value in loc:
        if isinstance(name, str) and name[0] == '_':
            continue
        if isinstance(value, Tensor):
            ou[name] = tuple(value.size())# + (value.grad_fn is not None,)
        elif isinstance(value, list) and any(isinstance(x, Tensor) for x in value):
            as_dict = watch([(ii, value[ii]) for ii in range(len(value))])
            as_list = [None] * (max(as_dict.keys()) + 1)
            for ii in as_dict.keys():
                as_list[ii] = as_dict[ii]
            ou[name] = as_list
        elif isinstance(value, dict):
            ou[name] = watch([(key, value[key]) for key in value])

    return dict(sorted(ou.items()))

def str_clean(string, *args):
    for arg in args:
        string = string.replace(arg,'')
    return string