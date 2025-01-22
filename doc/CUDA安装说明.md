# 内容
在 Ubuntu 22+ 系统下安装AI检测系统所需要的cuda环境

# 步骤

1. 安装**CUDA 11.8**
   - 运行命令: ` sudo sh cuda_11.8.0_520.61.05_linux.run --override `
   - 选择框中选择安装**CUDA Toolkit**，如果系统已经安装nvida的兼容驱动就不要选择安装**driver**
   - 在文件 **/etc/ld.so.conf** 中添加如下一行内容: **/usr/local/cuda-11.8/lib64**，可参考上一步命令的输出
   - 运行命令: ` sudo ldconfig `
2. 安装**cuDNN 8.9.7 for CUDA 11.x**, 执行如下命令: 
   -  ` sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb `
   -  ` sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-*-keyring.gpg /usr/share/keyrings/ `，可参考上一步命令的输出
   -  ` sudo apt update `
   -  ` sudo apt install libcudnn8 `