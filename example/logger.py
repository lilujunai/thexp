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

import time
from thexp.frame.logger import Logger
from thexp.frame.meter import Meter
from thexp.frame.params import Params
logger = Logger()




meter = Meter()
meter.a = 3.13524635465
meter.b = 5.13524635465
meter.c = 5.13524635465
meter.d = 5.13524635465
meter.e = 5.13524635465
meter.f = 5.13524635465
meter.f1 = 5.13524635465
meter.f2 = 5.13524635465
meter.f3 = 5.13524635465

# logger.info("\n",meter)

for i in range(10):
    logger.inline(meter)
    time.sleep(0.2)