# -*- encoding: utf-8 -*-
"""
日志配置模块.

@Time    :   2024/03/22 14:25:39
@Author  :   creativor 
@Version :   1.0
"""

import sys
from loguru import logger


def __init():
    # TODO: 从配置文件加载日志配置
    logger.configure(
        handlers=[
            {
                "sink": "log/ai-detection-{time:YYYY-MM-DD}.log",  # 自动在文件名中添加日期
                "level": "INFO",
                "rotation": "00:00",  # 每天午夜轮转生成新的日志文件
                "enqueue": True,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> - [{process.name}({process}):{thread.name}] - <cyan>{message}</cyan>",
            },
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": "{time:YYYY-MM-DD HH:mm:ss} <level>{level}</level> - [{process.name}({process}):{thread.name}] - <cyan>{message}</cyan>",
            },
        ]
    )


__init()