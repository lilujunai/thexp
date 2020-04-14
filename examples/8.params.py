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

from thexp import Params
params = Params()
params.epoch = 400
params.batch_size = 25
print(params)


from thexp import Params
class BaseParams(Params):

    def __init__(self):
        super().__init__()
        self.batch_size = 50
        self.topk = (1,2,3,4)
        self.optim_option(lr=0.009,moment=0.9)
        # 显示声明变量可以使用：
        # self.optim = self.optim_option(lr=0.009,moment=0.9)

params = BaseParams()
print(params)


from thexp import Params
params = Params()
params.from_args()
print(params)


from thexp.frame.params import BaseParams