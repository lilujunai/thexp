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
from collections import OrderedDict
from typing import Any

import torch

from ..base_classes.trickitems import AvgItem, NoneItem


class Meter:
    """
    m = Meter()
    m.short()
    m.int()
    m.float()
    m.percent()
    ...

    m.k += 10
    m.k.update()

    m.k.backstep()

    p = Param()
        list/tensor/optim
    """

    def __init__(self):
        self._param_dict = OrderedDict()
        self._format_dict = dict()
        self._convert_type = []

    def int(self, item):
        self._format_dict[item] = lambda x: "{:.0f}".format(x)

    def float(self, item, acc=4):
        self._format_dict[item] = lambda x: "{{:.{}f}}".format(acc).format(x)

    def percent(self, item, acc=2):
        self._format_dict[item] = lambda x: "{{:.{}%}}".format(acc).format(x)

    def tensorfloat(self, item, acc=4):
        def func(x: torch.Tensor):
            if len(x.shape) == 0:
                return "{{:.{}f}}".format(acc).format(x)
            else:
                return "{{:.{}f}}".format(acc).format(x.item())

        self._format_dict[item] = func

    def str_in_line(self, item):
        def func(x):
            l = str(x).split("\n")
            if len(l) > 1:
                return "{}...".format(l[0])
            else:
                return l[0]

        self._format_dict[item] = func

    def add_format_type(self, type, func):
        self._convert_type.append((type, func))

    def _convert(self, val):
        if type(val) in {int, float, bool, str}:
            return val
        # elif isinstance(val, torch.Tensor):
        #     if len(val.shape) == 1 and val.shape[0] == 1:
        #         return val[0]
        for tp, func in self._convert_type:
            if type(val) == tp:
                val = func(val)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            assert False
        else:
            value = self._convert(value)
            if isinstance(value, int) and name not in self._format_dict:
                self.int(name)
            elif isinstance(value, float) and name not in self._format_dict:
                self.float(name)
            elif isinstance(value, torch.Tensor) and name not in self._format_dict:
                if len(value.shape) == 0 or (sum(value.shape) == 1):
                    self.tensorfloat(name)
                else:
                    self.str_in_line(name)

            self._param_dict[name] = value

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __getattr__(self, item):
        if item.endswith("_"):
            return item.rstrip("_")

        if item not in self._param_dict:
            return NoneItem()
        else:
            return self._param_dict[item]

    def __getitem__(self, item):
        item = str(item)
        return self.__getattr__(item)

    def __repr__(self):
        log_dict = OrderedDict()
        for k, v in self._param_dict.items():
            if k in self._format_dict:
                v = self._format_dict[k](v)
            log_dict[k] = v

        return "  ".join(["@{}={}".format(k, v) for k, v in log_dict.items()])


class AvgMeter(Meter):
    def __init__(self):
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            assert False
        else:
            value = self._convert(value)
            if isinstance(value, int) and name not in self._format_dict:
                self.int(name)
            elif isinstance(value, float) and name not in self._format_dict:
                self.float(name)

            if isinstance(value, (float)):
                if name not in self._param_dict:
                    self._param_dict[name] = AvgItem()
                self._param_dict[name].update(value)
            elif isinstance(value, torch.Tensor):

                if len(value.shape) == 0 or sum(value.shape) == 1:
                    if name not in self._param_dict:
                        self._param_dict[name] = AvgItem()
                    self._param_dict[name].update(value)
                else:
                    self._param_dict[name] = value
            else:
                self._param_dict[name] = value

    def update(self, meter):
        for k, v in meter._param_dict.items():
            self[k] = v
        self._format_dict.update(meter._format_dict)
        self._convert_type.extend(meter._convert_type)

    def __repr__(self):
        log_dict = OrderedDict()
        for k, v in self._param_dict.items():
            if k in self._format_dict:
                if isinstance(v,AvgItem):
                    v = "{}({})".format(self._format_dict[k](v._item),self._format_dict[k](v.avg))
                else:
                    v = self._format_dict[k](v)
            log_dict[k] = v

        return "  ".join(["@{}={}".format(k, v) for k, v in log_dict.items()])