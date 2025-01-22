# -*- encoding: utf-8 -*-
"""
应用-初始化-模块.

@Time    :   2024/04/28 14:26:06
@Author  :   creativor 
@Version :   1.0
"""
import multiprocessing
import threading

__initialized = False
"""是否已经初始化，这是一个模块级别的变量"""
__lock = threading.Lock()
"""Create a lock to protect the initialization status."""


def __do_init__():
    """执行初始化应用"""
    #pytorch不支持fork启动方式，参考: https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    multiprocessing.set_start_method("spawn")
    # 初始化应用配置
    import log_config
    import config
    import detection
    import face.facefunction
    import mods.flask_loader
    import face_detection
    import myclip.clip_detection

    process_share = multiprocessing.Value("i", 0)
    # 初始化detection模块
    detection.initialize(process_share)
    # 初始化facefunction
    face.facefunction.initialize(process_share)
    # 初始化face_detection
    face_detection.initialize(process_share)
    # 初始化clip_detection
    myclip.clip_detection.initialize(process_share)
    # 加载 Flask server 子进程
    mods.flask_loader.initialize(process_share)
    


def app_startup():
    """应用程序的启动入口,本函数仅仅在应用启动时候执行一次"""
    global __initialized
    with __lock:
        if __initialized is False:
            # 执行初始化操作
            __do_init__()
            __initialized = True


# if platform.system() != "Windows":
#     # 非Windows系统使用os.fork()运行子进程，因此不会重复导入本模块，故在导入本模块时候初始化本应用
#     app_startup()
