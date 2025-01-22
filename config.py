# -*- encoding: utf-8 -*-
"""
应用配置模块.

顶层的配置项会被动态绑定到本模块的属性

@Time    :   2024/03/22 14:26:06
@Author  :   creativor 
@Version :   1.0
"""

import sys
import toml
from loguru import logger

__CONFIG_FILE = "config.toml"


def __init():
    # 读取 TOML 配置文件
    with open(__CONFIG_FILE, mode="r", encoding="utf-8") as file:
        configs = toml.load(file)
        if configs is None:
            logger.warning(f"配置文件{__CONFIG_FILE}内容为空")
            return
        # 获取当前模块对象
        current_module = sys.modules[__name__]
        # 将 dict 的属性动态绑定到模块的属性
        for key, value in configs.items():
            setattr(current_module,key,value)


__init()
