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

import torch

from thexp.calculate import accuracy as acc


def test_classify():
    labels = torch.tensor([0, 1, 2, 3])
    preds = torch.tensor([[5, 4, 3, 2], [5, 4, 3, 2], [5, 4, 3, 2], [5, 4, 3, 2]], dtype=torch.float)
    total, res = acc.classify(preds, labels, topk=(1, 2, 3, 4))
    assert total == 4
    assert res[0] == 1 and res[1] == 2 and res[2] == 3 and res[3] == 4,str(res)
