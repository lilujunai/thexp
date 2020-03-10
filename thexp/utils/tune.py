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

import numpy as np
import torch
import random

def fix_seed(self, seed=10):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    # np.random.seed(10)
    # print(np.random.get_state())
    # print(np.random.get_state()[1][0],np.random.rand(1))
    # print(np.random.get_state()[1][0],np.random.rand(1))
    # print(np.random.get_state()[1][0],np.random.rand(1))
    cur_seed = np.random.get_state()
    print(np.random.get_state()[1][0],np.random.rand(1))
    print(np.random.get_state()[1][0],np.random.rand(1))
    print(np.random.get_state()[1][0],np.random.rand(1))
    np.random.set_state(cur_seed)
    print(np.random.get_state()[1][0],np.random.rand(1))
    print(np.random.get_state()[1][0],np.random.rand(1))
    print(np.random.get_state()[1][0],np.random.rand(1))
