# -*- encoding: utf-8 -*-
"""
模型-配置-模块.

@Time    :   2024/05/14 17:24:24
@Author  :   creativor 
@Version :   1.0
"""
import functools
import json
import os

MODELS_DIR = "models"
"""模型目录"""
__CONFIG_FILE = "models/models.json"


@functools.lru_cache
def load_config():
    """
    加载模型配置.

    Returs:
         模型-配置-列表 (Cached)
    """
    if os.path.exists(__CONFIG_FILE):
        with open(__CONFIG_FILE, mode="r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def delete(model_name):
    """
    删除模型配置.会删除模型文件,同时清理缓存.

    Args:
        model_name (str): 模型名称
    """
    config = load_config()
    model = config.get(model_name)
    if model is None:
        return
    # 删除模型文件
    model_path = model["path"]
    if os.path.exists(model_path):
        os.remove(model_path)
    del config[model_name]
    save_config(config)
    # 清理缓存
    load_config.cache_clear()


def save_config(config):
    """
    保存模型配置. 同时清理缓存.

    """
    with open(__CONFIG_FILE, mode="w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        # 清理缓存
        load_config.cache_clear()
