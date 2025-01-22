# -*- encoding: utf-8 -*-
"""
自定义MQTT-客户端模块.

@Time    :   2024/03/28 14:26:06
@Author  :   creativor 
@Version :   1.0
"""
import threading
import paho.mqtt.client as mqtt
import config
from loguru import logger


_config = config.mqtt


class MQTTClient:
    """
    自定义的MQTT客户端
    """

    client: mqtt.Client

    def __init__(self, on_message: mqtt.CallbackOnMessage, thread):
        """初始化MQTT客户端

        Args:
            on_message (mqtt.CallbackOnMessage): 消息回调函数
            thread (DetectionThread): 检测线程实例
        """
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(
            username=_config["user_name"], password=_config["pass_word"]
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = on_message
        self.client.user_data_set(thread)
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.client.connect(_config["server_ip"], _config["port"], 60)
        with self.lock:
            if not self.running:
                self.client.loop_start()
                self.running = True

    def stop(self):
        with self.lock:
            if self.running:
                self.client.loop_stop()
                self.client.disconnect()
                self.running = False

    def on_connect(self, client: mqtt.Client, thread, flags, rc, properties=None):
        """连接到MQTT Broker事件

        Args:
            client (mqtt.Client): MQTT客户端
            thread (DetectionThread): 检测线程实例
        """
        copied_dict = dict(_config)
        del copied_dict["pass_word"]
        logger.info(
            f"Connected to MQTT server [{copied_dict}] with result code " + str(rc)
        )
        thread_subscribes = (
            thread.config["SUBSCRIBES"]
            if "SUBSCRIBES" in thread.config
            else _config["subscribe"]
        )
        subscribe_list = thread_subscribes.split(",")
        topic_list = [(x, 0) for x in subscribe_list]
        client.subscribe(topic_list)
