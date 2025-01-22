#!/usr/bin/env bash

### Linux shut down 脚本 ###
## by fszhang 2024-05-08 ##

ME=$(realpath $0)
BASE_DIR=$(dirname ${ME})
cd ${BASE_DIR}

PID_FILE=".pid"

PREVIOUS_PID=$(cat ${PID_FILE})

if [ -z ${PREVIOUS_PID} ]; then
    echo 进程ID为空,返回...
    exit
fi

if [ ! -d /proc/${PREVIOUS_PID} ]; then
    echo 进程[${PREVIOUS_PID}]并不存在,请检查运行环境...
    exit
fi

## 杀掉本进程及其子进程
kill_processes(){
    echo 关闭进程[${PREVIOUS_PID}]及其子进程...
    pkill -P ${PREVIOUS_PID}
    kill -9 ${PREVIOUS_PID}
}

echo 进程[${PREVIOUS_PID}]的信息如下:
ps ${PREVIOUS_PID}
echo "----"
read -p "是否关闭该进程及其子进程？请输入[y/n]:" input
case ${input} in
[yY][eE][sS] | [yY])
    kill_processes
    ;;
*)
    echo "退出..."
    exit 1
    ;;
esac
