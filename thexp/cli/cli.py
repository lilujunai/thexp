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
from ..frame.experiment import ExperimentViewer
import os


def clear_exp(exps_fn=None):
    i = ""
    while i.lower() != "y" or i.lower() != "n":
        i = input("really?[y/n]")

    if i == "y":
        if exps_fn is None:
            ExperimentViewer().drop()
        else:
            dir,fn = os.path.split(exps_fn)
            ExperimentViewer(dir,fn).drop()
        print("droped.")
    else:
        print("canceled.")

"""
 - 显示当前目录下所有的 exp
"""