# -*- coding:utf-8 -*-
# @FileName : log_util.py
# @Time : 2024/1/22 18:36
# @Author :fiv


import logging
import os
import pathlib
import time
from typing import Tuple

log = None
file_path = None


def get_log(filename=None) -> Tuple[logging.Logger, str]:
    global log
    global file_path
    if log is not None and file_path is not None:
        return log, file_path
    log = logging.getLogger(__name__)
    dir_path = pathlib.Path(__file__).parent.parent
    if filename is None:
        filename = "log"
    t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    filename = f"{filename}_{t}.log"
    file_path = os.path.join(dir_path, "log", filename)
    if not os.path.exists(os.path.join(dir_path, "log")):
        os.makedirs(os.path.join(dir_path, "log"))

    logging.basicConfig(
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:  %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=file_path, encoding="utf-8"),
        ],
        level=logging.INFO
    )
    log.info(f"Log file path: {file_path}")
    return log, file_path
