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
import json
import os
import re

import torch

from ..utils.generel_util import listdir_by_time


class Saver:
    _ckpt_fn_templete = "{}{:06d}.ckpt"
    _model_fn_templete = "model.{}{:06d}.pth"
    re_fn = re.compile("^[0-9]{7}\.ckpt$")
    re_keep_fn = re.compile("^keep.[0-9]{7}\.ckpt$")

    def __init__(self, ckpt_dir, max_to_keep=3, ):
        self.info = {}
        self.ckpt_dir = ckpt_dir
        self.max_to_keep = max_to_keep
        os.makedirs(ckpt_dir, exist_ok=True)

    def _build_checkpoint_name(self, epoch, replacement: bool, lasting):
        i = 0

        def build_fn():
            fn = Saver._ckpt_fn_templete.format(i, epoch)
            if lasting:
                fn = "keep.{}".format(fn)
            return fn

        absfn = os.path.join(self.ckpt_dir, build_fn())
        while replacement == False and os.path.exists(absfn):
            i += 1
            if i >= 9:
                if lasting:
                    kfns = self.find_keypoints()
                else:
                    kfns = self.find_checkpoints()
                return kfns[-1]

            absfn = os.path.join(self.ckpt_dir, build_fn())
        return absfn

    def _build_model_name(self, epoch):
        fn = os.path.join(self.ckpt_dir, Saver._model_fn_templete.format(0, epoch))
        return fn

    def find_models(self):
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        fs = [i for i in fs if i.endswith(".pth")]
        return fs

    def find_keypoint(self, epoch, all=False):
        """
        find all "keeped" checkpoint saved in the save dir.
        :return:
        """
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        re_fn = re.compile("^keep.[0-9]{:06d}\.ckpt$".format(epoch))
        fs = [os.path.join(self.ckpt_dir, i) for i in fs if re.search(re_fn, i) is not None]
        if all:
            return fs
        if len(fs) > 0:
            return fs[0]  # 返回最后保存的那一个
        return None

    def find_keypoint_info(self, epoch, all=False):
        fns = self.find_keypoint(epoch, all)
        if all:
            return ["{}.json".format(fn) for fn in fns]
        else:
            if fns is None:
                return None
            return "{}.json".format(fns)

    def find_keypoints(self):
        """
        find all "keeped" checkpoint saved in the save dir.
        :return:
        """
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        fs = [os.path.join(self.ckpt_dir, i) for i in fs if re.search(Saver.re_keep_fn, i) is not None]
        return fs

    def find_checkpoint(self, epoch, all=False):
        """
        find checkpoints saved in [epoch]
        :param epoch:  int
        :param all: 是否返回多个如果发现了多个保存在同一个epoch的checkpoint,
            default False
        :return:  if 'all' is True, return all checkpoints in that epoch
            else, return the last created checkpoint
        """
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        re_fn = re.compile("^[0-9]{:06d}\.ckpt$".format(epoch))
        fs = [os.path.join(self.ckpt_dir, i) for i in fs if re.search(re_fn, i) is not None]
        if all:
            return fs
        if len(fs) > 0:
            return fs[0]  # 返回最后保存的那一个
        else:
            return None

    def find_checkpoints(self):
        """
        find all checkpoint saved in the save dir.
        :return:
        """
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        fs = [os.path.join(self.ckpt_dir, i) for i in fs if re.search(Saver.re_fn, i) is not None]
        return fs

    def find_checkpoint_info(self, epoch, all=False):
        fns = self.find_checkpoint(epoch, all)
        if all:
            return ["{}.json".format(fn) for fn in fns]
        else:
            if fns is None:
                return None
            return "{}.json".format(fns)

    def _check_max_checkpoint(self):
        fs = self.find_checkpoints()
        while len(fs) > self.max_to_keep:
            self.check_remove(fs[-1])
            fs.pop()

    def save_model(self, val, state_dict, extra_info: dict = None):
        if self._check_isint_or_str(val):
            fn = self._build_model_name(val)
        else:
            fn = val
        json_fn = "{}.json".format(fn)
        torch.save(state_dict, fn)
        with open(json_fn, "w", encoding="utf-8") as w:
            json.dump(extra_info, w, indent=2)

        return fn

    def save_checkpoint(self, val, state_dict, extra_info: dict = None, replacement: bool = False, lasting=False):
        if self._check_isint_or_str(val):
            fn = self._build_checkpoint_name(val, replacement, lasting)
        else:
            fn = val

        json_fn = "{}.json".format(fn)
        torch.save(state_dict, fn)
        with open(json_fn, "w", encoding="utf-8") as w:
            json.dump(extra_info, w, indent=2)
            self._check_max_checkpoint()

        return fn

    def save_keypoint(self, val, state_dict, extra_info: dict = None, replacement: bool = False):
        return self.save_checkpoint(val, state_dict, extra_info, replacement, True)

    def _check_isint_or_str(self, val):
        assert type(val) in {int, str}
        if isinstance(val, int):
            return True
        return False

    def load_model(self, val):
        if self._check_isint_or_str(val):
            fn = self._build_model_name(val)
        else:
            fn = val
        return torch.load(fn)

    def load_model_info(self, val):
        if self._check_isint_or_str(val):
            fn = self._build_model_name(val)
        else:
            fn = val
        json_fn = "{}.json".format(fn)
        with open(json_fn, "r", encoding="utf-8") as r:
            return json.load(r)

    def load_keypoint(self, val):
        """

        :param val: epoch number[int] or file path[str]
        :return:  None or torch.load() result
        """
        if self._check_isint_or_str(val):
            fn = self.find_keypoint(val)
        else:
            fn = val
        if fn is not None and os.path.exists(fn):
            return torch.load(fn)
        return None

    def load_keypoint_info(self, val):
        """

        :param val: epoch number[int] or file path[str]
        :return:  None or torch.load() result
        """
        if self._check_isint_or_str(val):
            fn = self.find_keypoint(val)
        else:
            fn = val
        if fn is not None:
            json_fn = "{}.json".format(fn)
            with open(json_fn, "r", encoding="utf-8") as r:
                return json.load(r)
        return None

    def load_checkpoint(self, val, byindex=False):
        """

        :param val: epoch number[int] or file path[str]
        :return:  None or torch.load() result
        """
        assert not (not isinstance(val, int) and byindex), "If byindex = True, val must be an index(int)."
        if byindex:
            fn = self.find_checkpoints()[val]
        elif self._check_isint_or_str(val):
            fn = self.find_checkpoint(val)
        else:
            fn = val
        if fn is not None and os.path.exists(fn):
            return torch.load(fn)
        return None

    def load_checkpoint_info(self, val, byindex=False):
        assert not (not isinstance(val, int) and byindex), "If byindex = True, val must be an index(int)."
        if byindex:
            fn = self.find_checkpoints()[val]
        elif self._check_isint_or_str(val):
            fn = self.find_checkpoint(val)
        else:
            fn = val
        if fn is not None:
            json_fn = "{}.json".format(fn)
            with open(json_fn, "r", encoding="utf-8") as r:
                return json.load(r)
        return None

    def check_remove(self, fn, with_json=True):
        if os.path.exists(fn):
            os.remove(fn)
        if with_json:
            jfn = "{}.json".format(fn)
            if os.path.exists(jfn):
                os.remove(jfn)

    def clear_models(self):
        fns = self.find_models()
        for i in fns:
            self.check_remove(i)

    def clear_checkpoints(self):
        for i in self.find_checkpoints():
            self.check_remove(i)

    def clear_keypoints(self):
        for i in self.find_keypoints():
            self.check_remove(i)

    def summary(self, detail=False):
        print("checkpoints:")
        print(" || ".join(self.find_checkpoints()))
        print("keppoints:")
        print(" || ".join(self.find_keypoints()))
        print("models:")
        print(" || ".join(self.find_models()))
