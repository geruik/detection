# -*- encoding: utf-8 -*-
"""
一个用于管理和启动 Streamlit 应用的工具，旨在简化通过命令行启动和激活 Streamlit 应用的过程.

@Time    :   2024/12/03 11:07:06
@Author  :   creativor 
@Version :   1.0
"""
import base64
import asyncio
import time
import websockets
import sys


DEFAULT_SERVER_ENDPOINT = "127.0.0.1:8505"
"""
默认 WebSocket 服务器地址
"""


async def send_data(ws_url, base64_data):
    """发送数据到 WebSocket 服务器"""
    # Base64 解码
    decoded_data = base64.b64decode(base64_data)

    # 连接到 WebSocket 服务器
    async with websockets.connect(ws_url) as websocket:
        # 发送二进制数据
        await websocket.send(decoded_data)
        print(f"Sent: {decoded_data}")

        # 可选：接收 WebSocket 服务器的响应
        response = await websocket.recv()
        print(f"Received: {response}")


def main():
    """程序入口。第一个命令行参数是 WebSocket 服务器地址"""
    # Base64 编码的空白请求数据
    base64_data = "WggKABIAGgAiAA=="

    server_endpoint = DEFAULT_SERVER_ENDPOINT

    # 获取所有命令行参数
    args = sys.argv
    # 获取特定参数
    if len(args) > 1:
        server_endpoint = args[1]

    # WebSocket 服务器地址
    ws_url = f"ws://{server_endpoint}/_stcore/stream"

    # 调用发送数据的函数第1次
    asyncio.get_event_loop().run_until_complete(send_data(ws_url, base64_data))

    time.sleep(2)

    # 调用发送数据的函数第2次
    asyncio.get_event_loop().run_until_complete(send_data(ws_url, base64_data))
    
    print("完成激活.")


if __name__ == "__main__":
    main()
