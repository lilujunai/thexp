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
import pprint
import time
import warnings

from thexp import Meter


class TimeIt:
    def __init__(self):
        self.last_update = None
        self.ends = False
        self.times = []

    def offset(self):
        now = time.time()

        if self.last_update is None:
            offset = 0
        else:
            offset = now - self.last_update

        self.last_update = now
        return offset, now

    def clear(self):
        self.last_update = None
        self.ends = False
        self.times.clear()

    def start(self):
        self.clear()
        self.mark("start")

    def mark(self, key):
        if self.ends:
            warnings.warn("called end method, please use start to restart timeit")
            return
        key = str(key)
        offset, now = self.offset()
        self.times.append((key, offset, now))

    def end(self):
        self.ends = True
        self.mark("end")

    def meter(self):
        meter = Meter()
        for key, offset, _ in self.times:
            meter[key] = offset
        return meter

    def __str__(self):
        return pprint.pformat(self.times)


timeit = TimeIt()
