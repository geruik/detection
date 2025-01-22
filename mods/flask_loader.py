# -*- encoding: utf-8 -*-
"""
Flask Server - 加载器.

@Time    :   2024/03/28 14:26:06
@Author  :   creativor 
@Version :   1.0
"""

import multiprocessing


def load_flask_server(*args, **kwargs):
    """加载Flask API server"""
    import flask_api
    # 初始化flask
    flask_api.start(kwargs["process_share"])

def initialize(process_share):
    """初始化本模块"""
    # 创建一个子进程，并在子进程中启动 Flask 服务器
    flask_process = multiprocessing.Process(target=load_flask_server, kwargs={"process_share": process_share})
    flask_process.start() 
