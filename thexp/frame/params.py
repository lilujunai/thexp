"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""
import pprint as pp
from collections import OrderedDict
from typing import Any

import fire
import torch

class AttrDict(dict):
    def __getattr__(self, item):
        return self[item]


class BaseParams:
    def __init__(self):
        self._param_dict = OrderedDict()
        self._exp_name = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._param_dict[name] = value

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)


    def __getattr__(self, item):
        if item not in self._param_dict:
            raise AttributeError(item)
        return self._param_dict[item]

    def __getitem__(self, item):
        return self._param_dict[item]

    def __repr__(self):
        return "{}".format(self.__class__.__name__) + pp.pformat([(k, v) for k, v in self._param_dict.items()])

    def _can_in_dir_name(self, obj):
        for i in [int, float, str, bool]:
            if isinstance(obj, i):
                return True
        if isinstance(obj, torch.Tensor):
            if len(obj.shape) == 0:
                return True
        return False

    def build_exp_name(self, *names, prefix="", sep="_", ignore_mode="add"):
        prefix = prefix.strip()
        res = []
        if len(prefix) != 0:
            res.append(prefix)
        if ignore_mode == "add":
            for name in names:
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if self._can_in_dir_name(obj):
                        res.append("{}={}".format(name, obj))
                else:
                    res.append(name)

        elif ignore_mode == "del":
            for name in names:
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if self._can_in_dir_name(obj):
                        res.append("{}={}".format(name, obj))
        else:
            assert False

        self._exp_name = sep.join(res)
        return self._exp_name

    def get_exp_name(self):
        assert self._exp_name is not None, "run build_exp_name() before get_exp_name()"
        return self._exp_name

    def from_args(self):
        def func(**kwargs):
            for k, v in kwargs.items():
                cur = self
                ks = k.split(".")
                for ka in ks[:-1]:
                    if ka not in cur._param_dict:
                        cur[ka] = AttrDict()
                    cur = cur[ka]
                cur[ks[-1]] = v

        fire.Fire(func)

    def optim_option(self,lr=0.001,**kwargs):
        """
        创建优化器的参数，添加后会在实例中添加 "optim" 变量，包含传入的各个参数，在使用时：

        self.optim_option(lr=0.001,moment=0.9,...)
        SGD(param=...,**param.optim)
        """
        optim = AttrDict()
        optim["lr"] = lr
        for k,v in kwargs.items():
            optim[k] = v
        self.optim = optim
        return optim


class Params(BaseParams):
    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.eidx = 1
        self.idx = 0
        self.global_step = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"





if __name__ == '__main__':
    print(Params())
