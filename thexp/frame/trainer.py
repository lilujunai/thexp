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
import bisect
import os
import pprint as pp
import warnings
from functools import wraps
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .databundler import DataBundler
from .experiment import Experiment
from .logger import Logger
from .meter import AvgMeter
from .params import Params
from .plotter import Plotter
from .rndmanager import RndManager
from .saver import Saver


class BaseTrainer:
    _ignore_call_back = {"model_dict", "optim_dict",
                         "model_state_dict", "optim_state_dict",
                         "create_checkpoint_dict", "create_checkpoint_dict",
                         "iter_train_dataloader", "iter_eval_dataloader", "iter_test_dataloader",
                         "predict", "preprocess",
                         "add_callback"}

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)

        def wrapper(func, _call_set: list):
            @wraps(func)
            def _newfunc(*args, **kwargs):
                for callback in _call_set:
                    if callback.enable:
                        callback.on_begin(self, func, self.params, *args, **kwargs)
                try:
                    _meter = func(*args, **kwargs)
                except BaseException as e:
                    _handles = [callback.on_exception(self, func, self.params, e, *args, **kwargs)
                                for callback in _call_set]

                    print(_handles)
                    if any(_handles):
                        return None
                    else:
                        raise e

                for callback in _call_set:
                    if callback.enable:
                        callback.on_end(self, func, self.params, _meter, *args, **kwargs)
                return _meter

            return _newfunc

        self._callback_set = []
        self._callback_name_set = set()

        vars = dir(self)
        for name in vars:
            value = getattr(self, name)
            if name.startswith("_"):
                continue
            if callable(value):
                setattr(self, name, wrapper(value, self._callback_set))
        return self

    def __init__(self, params: Params):
        self.model_dict = {}
        self.optim_dict = {}
        self.databundler_dict = {}
        self.params = params
        self.train_epoch_toggle = False
        self.train_toggle = False
        self.device = torch.device(params.device)

    def initial_callback(self):
        pass

    def initial_trainer(self, params: Params):
        pass

    def initial_exp(self, exps_dir):
        exp_dir = os.path.join(exps_dir, self.params.get_exp_name())
        self.experiment = Experiment(exp_dir)
        self.experiment.start_exp()
        self.logger = Logger()
        self.logger.add_log_dir(self.experiment.hold_exp_part("logs", exts=[".log"]))
        self.saver = Saver(self.experiment.hold_exp_part("modules", exts=[".pth", ".json"]))
        self.plotter = Plotter(self.experiment.hold_exp_part("plot", exts=[".eps", ".jpeg", ".jpg",
                                                                           ".pdf", ".png", ".svg", ".tif", ".tiff"]))
        self.rnd = RndManager(self.experiment.hold_exp_part("rnd", exts=[".rnd"]))
        self.experiment.add_event_listener(self.plotter.dynamic_board, exts=[".bd"])
        self.writter = SummaryWriter(self.experiment.hold_exp_part("board", exts=[".bd"]), filename_suffix=".bd")
        self.initial_callback()
        self.initial_trainer(self.params)

    def train(self):
        params = self.params
        for eidx in range(params.eidx, params.epoch + 1):
            self.train_epoch(params.eidx, params)
            params.eidx = eidx + 1
            if self.train_toggle:
                self.train_toggle = False
                break

    def train_epoch(self, eidx, params):
        avgMeter = AvgMeter()
        for idx, batch_data in enumerate(self.iter_train_dataloader()):
            self.change_mode(True)
            meter = self.train_batch(eidx, idx, self.params.global_step, batch_data, params, self.device)
            avgMeter.update(meter)
            self.change_mode(False)

            params.global_step += 1
            params.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

        return avgMeter

    def train_step(self, steps):
        param = self.params
        i = 0
        while steps > 0:
            for idx, data in enumerate(self.iter_train_dataloader()):
                meter = self.train_batch(0, idx, i, data, param, self.device)
                steps -= 1
                if steps <= 0:
                    return meter

    def train_on_batch(self, batch_data):
        self.train_batch(0, 0, 0, batch_data, self.params, self.device)

    def test(self):
        loader = self.iter_test_dataloader()
        if loader is None:
            self.logger.info("Have no test dataset, ignored test.")
            return None
        return self.test_eval_logic(loader, self.params)

    def eval(self):
        loader = self.iter_eval_dataloader()
        if loader is None:
            self.logger.info("Have no eval dataset, ignored eval.")
            return None
        return self.test_eval_logic(loader, self.params)

    def _regist_databundler(self, key, val):
        assert isinstance(val, (DataBundler, DataLoader))
        if isinstance(val, DataLoader):
            val = DataBundler().add(val)
        self.databundler_dict[key] = val

    def regist_databundler(self, train=None, eval=None, test=None):
        if train is not None:
            self.regist_train_databundler(train)
        if eval is not None:
            self.regist_eval_databundler(eval)
        if test is not None:
            self.regist_test_databundler(test)

    def regist_eval_databundler(self, eval):
        self._regist_databundler("eval", eval)

    def regist_test_databundler(self, test):
        self._regist_databundler("tests", test)

    def regist_train_databundler(self, train):
        self._regist_databundler("train", train)

    def iter_train_dataloader(self) -> DataBundler:
        return self.databundler_dict.get("train",None)

    def iter_eval_dataloader(self) -> DataBundler:
        return self.databundler_dict.get("eval",None)

    def iter_test_dataloader(self) -> DataBundler:
        return self.databundler_dict.get("tests",None)

    def _load_checkpoint_dict(self, res, strict, ignore_optim):
        if res is None:
            return False
        self.params.eidx = res["_eidx"]+1
        self.params.idx = res["_idx"]
        self.params.global_step = res["_global_step"]
        self.load_state_dict(res, strict)
        if not ignore_optim:
            self.load_optim_dict(res, strict)
        return True

    def load_keypoint(self, epoch, strict=True, ignore_optim=False):
        res = self.saver.load_keypoint(epoch)
        if self._load_checkpoint_dict(res, strict, ignore_optim):
            self.logger.info("load keypoint. current = {}/{}.".format(self.params.eidx,self.params.epoch))
            extra = self.saver.load_checkpoint_info(epoch)
            if extra is not None:
                self.logger.info("Extra info:")
                self.logger.info(pp.pformat(extra))
        else:
            self.logger.info("checkpoint of {} epoch not found. Choice from:".format(epoch))
            self.logger.info(self.saver.find_keypoints())

    def load_checkpoint(self, epoch, strict=True, byindex=False, ignore_optim=False, not_exist_ok=False):
        res = self.saver.load_checkpoint(epoch, byindex)
        if res is None and not not_exist_ok:
            self.logger.info("file of epoch {} not found, only have {}".format(epoch, self.saver.find_checkpoints()))
            raise FileNotFoundError()

        if self._load_checkpoint_dict(res, strict, ignore_optim):
            self.logger.info("checkpoint of {} epoch not found. Choice from:".format(epoch))
            self.logger.info(self.saver.find_checkpoints())
        else:
            self.logger.info("load checkpoint of {} epoch. ".format(self.params.eidx))
            extra = self.saver.load_checkpoint_info(epoch, byindex)
            if extra is not None:
                self.logger.info("Extra info:")
                self.logger.info(pp.pformat(extra))

    def load_model(self, val, strict=True):
        self.load_state_dict(self.saver.load_model(val), strict)

    def save_keypoint(self, extra_info=None, replacement=False):
        state_dict = self.checkpoint_dict()
        fn = self.saver.save_keypoint(state_dict["_eidx"], state_dict, extra_info, replacement)
        self.logger.info("save keypoint in {}".format(fn))
        return fn

    def save_checkpoint(self, extra_info=None, replacement=False):
        state_dict = self.checkpoint_dict()
        fn = self.saver.save_checkpoint(state_dict["_eidx"], state_dict, extra_info, replacement)
        self.logger.info("save checkpoint in {}".format(fn))
        return fn

    def save_model(self, extra_info=None):
        state_dict = self.state_dict()
        fn = self.saver.save_model(self.params.eidx, state_dict, extra_info)
        self.logger.info("save model in {}".format(fn))
        return fn

    def add_callback(self, callback):
        """
        添加一个回调函数
        :type callable,str
        :param callback:
        :return:
        """
        msg = None
        cb_name = callback.__class__.__name__
        if callback not in self._callback_set and cb_name in self._callback_name_set:
            msg = "Callback duplicate."
            callback.on_hook_failed(self, msg)

        if msg is not None:
            return False

        bisect.insort(self._callback_set, callback)
        self._callback_name_set.add(cb_name)
        callback.on_hooked(self, self.params)
        return True

    def remove_callback(self, callback):
        msg = None
        if callback not in self._callback_set:
            return False

        cb_name = callback.__class__.__name__
        self._callback_set.remove(callback)
        self._callback_name_set.remove(cb_name)
        return True

    '''module和optim的一部分方法集成'''

    def load_state_dict(self, state_dict, strict=True):
        for k in self.model_dict:
            if k in state_dict:
                self.model_dict[k].load_state_dict(state_dict[k], strict=strict)
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))

    def load_optim_dict(self, state_dict, strict=True):
        for k in self.optim_dict:
            if k in state_dict:
                self.optim_dict[k].load_state_dict(state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))

    def load_one_state_dict(self, name, state_dict, strict=True):
        self.model_dict[name].load_state_dict(state_dict, strict)

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.model_dict.items()}

    def one_state_dict(self, name):
        return self.model_dict[name].state_dict()

    def checkpoint_dict(self):
        res = self.state_dict()
        res.update(self.optim_state_dict())
        res["_eidx"] = self.params.eidx
        res["_idx"] = self.params.idx
        res["_global_step"] = self.params.global_step
        return res

    def optim_state_dict(self):
        return {k: v.state_dict() for k, v in self.optim_dict.items()}

    def change_mode(self, train=True):
        for k, v in self.model_dict.items():
            if train:
                v.train()
            else:
                v.eval()

    def to(self, device):
        for k, v in self.model_dict.items():
            self.__setattr__(k, v.to(device))

    '''magic functions'''

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if isinstance(value, torch.device):
            pass
        elif isinstance(value, torch.nn.Module):
            self.model_dict[name] = value
        elif isinstance(value, Optimizer):
            self.optim_dict[name] = value

    # need to reimplement

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        raise NotImplementedError()

    def test_eval_logic(self, dataloader, param: Params):
        raise NotImplementedError()

    def predict(self, xs):
        raise NotImplementedError()

    def estimate_memory(self):
        for _, v in self.model_dict.items():
            pass


class Trainer(BaseTrainer):
    def initial_callback(self):
        from .callbacks import EvalCallback, LoggerCallback
        ec = EvalCallback(1, 5)
        ec.hook(self)

        lc = LoggerCallback()
        lc.hook(self)

    def initial_trainer(self, params: Params):
        pass

    def predict(self, xs):
        pass

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch.device):
        pass

    def test_eval_logic(self, dataloader, param: Params):
        pass




