#!/usr/bin/env bash

### Linux启动脚本 ###
## by fszhang 2022-07-08 ##

# python的命令名称, 默认是python
PYTHON=python

ME=$(realpath $0)
BASE_DIR=$(dirname ${ME})
cd ${BASE_DIR}
PID_FILE=".pid"
PORT_NUMBER=8505
#LOG_FILE=log/out.log

echo "启动智能AI助手中..."
# 启动智能AI助手, 由于经常报“[Errno 5] Input/output error”错误，因此把标准输出和错误输出重定向到空设备
${PYTHON} -m streamlit run webui.py --server.port ${PORT_NUMBER} >/dev/null 2>&1 &
PREVIOUS_PID=$!
echo ${PREVIOUS_PID} >${PID_FILE}
echo "完成启动智能AI助手, url-> http://127.0.0.1:${PORT_NUMBER}"
sleep  3s
echo "自动激活智能AI助手..."
${PYTHON} mods/streamlit_activator.py 127.0.0.1:${PORT_NUMBER}