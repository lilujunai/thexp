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

用于
"""

from ..frame import Params,Meter
import torch
from . import accuracy as acc

class classify_logits:
    def test_eval_logic(self, dataloader, param: Params):
        with torch.no_grad():
            count_dict = Meter()
            for xs, labels in dataloader:
                xs, labels = xs.to(self.device), labels.to(self.device)
                preds = self.predict(xs)
                total, topk_res = acc.classify(preds, labels, topk=param.topk)
                count_dict["total"] += total
                for i, topi_res in zip(param.topk, topk_res):
                    count_dict[i] += topi_res
        return count_dict
