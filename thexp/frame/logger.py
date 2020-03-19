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
from datetime import datetime
from typing import Any

from ..utils.generel_util import curent_date
from ..utils.screen import ScreenStr


class Logger:
    VERBOSE = 0
    V_DEBUG = -1
    V_INFO = 0
    V_WARN = 1
    V_ERROR = 2
    V_FATAL = 3
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        if Logger._instance is not None:
            return Logger._instance
        return super().__new__(cls)

    def __init__(self, adddate=True, datefmt: str = '%y-%m-%d %H:%M:%S', sep: str = " - ", stdout=True):
        """
        :param datefmt:
        :param sep:
        """
        if Logger._instance is not None:
            return

        self.adddate = adddate
        self.datefmt = datefmt
        self.out_channel = []
        self.pipe_key = set()
        self.sep = sep
        self.return_str = ""
        self.listener = []
        self.stdout = stdout
        Logger._instance = self

    def format(self, *values, prefix="", inline=False, fix=0):
        """根据初始化设置 格式化 前缀和LogMeter"""
        if self.adddate:
            cur_date = datetime.now().strftime(self.datefmt)
        else:
            cur_date = ""

        space = [cur_date, "{}".format(prefix), *["{}".format(str(i)) for i in values]]
        space = [i for i in space if len(i.strip()) != 0]

        if fix >= 0:
            left, right = self.sep.join(space[:fix + 1]), self.sep.join(space[fix + 1:])
            fix = len(left) + len(self.sep)
            logstr = self.sep.join((left, right))

            if inline:
                return "\r{}".format(logstr), fix
            else:
                return "{}\n".format(logstr), fix

        space = self.sep.join(space)

        if inline:
            return "\r{}".format(space), 0
        else:
            return "{}\n".format(space), 0

    def inline(self, *values, prefix="", fix=0):
        """在一行内输出 前缀 和 LogMeter"""
        logstr, fix = self.format(*values, prefix=prefix, inline=True, fix=fix)

        self.handle(logstr, fix=fix)

    def info(self, *values, prefix=""):
        """以行为单位输出 前缀 和 LogMeter"""
        logstr, fix = self.format(*values, prefix=prefix, inline=False)
        self.handle(logstr, level=Logger.V_INFO, fix=fix)

    def debug(self, *values, prefix=""):
        logstr, fix = self.format("DEBUG", *values, prefix=prefix, inline=False)
        self.handle(logstr, level=Logger.V_DEBUG, fix=fix)

    def warn(self, *values, prefix=""):
        logstr, fix = self.format("WARN", *values, prefix=prefix, inline=False)
        self.handle(logstr, level=Logger.V_WARN, fix=fix)

    def error(self, *values, prefix=""):
        logstr, fix = self.format("ERROR", *values, prefix=prefix, inline=False)
        self.handle(logstr, level=Logger.V_ERROR, fix=fix)

    def fatal(self, *values, prefix=""):
        logstr, fix = self.format("FATAL", *values, prefix=prefix, inline=False)
        self.handle(logstr, level=Logger.V_FATAL, fix=fix)

    def newline(self):
        """换行"""
        self.handle("\n")

    def handle(self, logstr, end="", level=0, **kwargs):
        """
        handle log stinrg，以指定的方式输出
        :param logstr:
        :param _:
        :param end:
        :return:
        """
        for listener in self.listener:
            listener(logstr, end, level)

        if level < Logger.VERBOSE:
            return

        if logstr.startswith("\r"):
            fix = kwargs.get("fix", 0)
            self.return_str = logstr
            self.print(ScreenStr(logstr, leftoffset=fix), end=end)
        else:
            if len(self.return_str) != 0:
                self.print(self.return_str, end="\n")
            self.print(logstr, end="")

            for i in self.out_channel:
                with open(i, "a", encoding="utf-8") as w:
                    if len(self.return_str) != 0:
                        w.write("{}\n".format(self.return_str.strip()))
                    w.write(logstr)

            self.return_str = ""

    def print(self, *args, end='\n'):
        if self.stdout:
            print(*args, end=end, flush=True)

    def enbale_stdout(self, val):
        self.stdout = val

    def add_log_dir(self, dir):
        """添加一个输出到文件的管道"""
        if dir in self.pipe_key:
            self.info("Add pipe {}, but already exists".format(dir))
            return False

        os.makedirs(dir, exist_ok=True)

        i = 0
        cur_date = curent_date(fmt="%y%m%d%H%M%S")
        fni = os.path.join(dir, "{}.{}.log".format(cur_date, i))
        while os.path.exists(fni):
            i += 1
            fni = os.path.join(dir, "{}.{}.log".format(cur_date, i))

        self.print("add output channel on {}".format(fni))
        self.out_channel.append(fni)
        self.pipe_key.add(dir)
        return True

    def add_log_listener(self, func):
        self.listener.append(func)
