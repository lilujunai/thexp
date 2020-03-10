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
import sys
sys.path.insert(0,"../")
from thexp import __VERSION__
print(__VERSION__)


import time
from thexp.frame import Logger,Saver,Experiment

exp = Experiment("./exp")

@exp.keycode()
def train():
    logger = Logger()
    logger.add_log_dir(exp.hold_exp_part("log",[".log"]))
    save = Saver(exp.hold_exp_part("save",[".ckpt",".pth"]))
    for i in range(10):
        for j in range(5):
            save.save_checkpoint(j, {}, {})
            time.sleep(0.2)
            logger.info(i)


exp.start_exp()
train()