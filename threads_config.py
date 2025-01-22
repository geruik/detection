# -*- encoding: utf-8 -*-
"""
任务-线程-配置-模块.

@Time    :   2024/03/22 14:24:24
@Author  :   creativor 
@Version :   1.0
"""
import functools
import json
import os
import uuid


__CONFIG_FILE = "threads_config.json"
__FACE_CONFIG_FILE = "face_threads_config.json"
__CLIP_CONFIG_FILE = "clip_threads_config.json"


def __base_load_config(path):
    if os.path.exists(path):
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def __base_save_config(config, path):
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


@functools.lru_cache
def load_config():
    """
    加载任务配置.

    Returs:
         任务-线程-列表 (Cached)
    """
    return __base_load_config(__CONFIG_FILE)


def save_config(config):
    """
    保存任务配置. 同时清理缓存.

    """
    __base_save_config(config, __CONFIG_FILE)
    # 清理缓存
    load_config.cache_clear()


"""人脸检测线程"""


@functools.lru_cache
def face_load_config():
    return __base_load_config(__FACE_CONFIG_FILE)


def face_save_config(config):
    __base_save_config(config, __FACE_CONFIG_FILE)
    # 清理缓存
    face_load_config.cache_clear()


@functools.lru_cache
def clip_load_config():
    return __base_load_config(__CLIP_CONFIG_FILE)


def clip_save_config(config):
    __base_save_config(config, __CLIP_CONFIG_FILE)
    # 清理缓存
    clip_load_config.cache_clear()


def is_model_used(model_name: str) -> bool:
    """判断指定模型是否已经被某个线程使用
    Args:
        model_name (str): 指定模型名字
    Returns:
        bool: True 如果指定模型已经被使用， 否则 False
    """
    config = load_config()
    for _, thread in config.items():
        if thread["MODEL_NAME"] == model_name:
            return True
    return False


def isSameNameExists(thread_name: str, thread_id: str, face=False, clip=False) -> bool:
    """判断指定线程名字的线程是否已经存在

    Args:
        thread_name (str): 指定线程名字
        thread_id (str): 原线程对应的线程id

    Returns:
        bool: True 如果同样名字且线程ID不同的线程存在， 否则 False
    """
    if face:
        config = face_load_config()
    elif clip:
        config = clip_load_config()
    else:
        config = load_config()
    exit_thread = config.get(thread_name)
    return (
        True if (exit_thread and exit_thread.get("THREAD_ID") != thread_id) else False
    )


def generate_thread_id() -> str:
    """生成新的线程ID

    Returns:
        str: UUID格式的线程id
    """
    return str(uuid.uuid4())
