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
import os
import pickle

from ..utils import random


class RndManager:
    """用于控制训练过程中的随机种子，保证数据集/"""
    STATE_DATESET = "dataset"  # 推荐在加载数据集前用，保证每次随机切分的数据集完全相同
    STATE_MODEL = "model"  # 推荐在初始化模型时候用，保证模型初始化参数完全相同
    STATE_TRAIN_BEGIN = "begin"  # 推荐在训练开始前用，保证每次训练的反向传播更新过程完全相同
    STATE_BREAK = "break"

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def save_rnd_state(self, name):
        stt = random.get_state()
        with open(self._build_state_name(name), "wb") as f:
            pickle.dump(stt, f)

    def have_rnd_state(self, name):
        return os.path.exists(self._build_state_name(name))

    def get_rnd_state(self, name):
        if not self.have_rnd_state(name):
            return None
        with open(self._build_state_name(name), "rb") as f:
            return pickle.load(f)

    def load_rnd_state(self, name, not_exist_save=False):
        stt = self.get_rnd_state(name)
        if stt is not None:
            random.set_state(stt)
            return True
        elif not_exist_save:
            self.save_rnd_state(name)
            return True
        return False

    def del_rnd_state(self, name):
        if self.have_rnd_state(name):
            os.remove(self._build_state_name(name))

    def _build_state_name(self, name):
        return os.path.join(self.save_dir, "{}.rnd".format(name))
