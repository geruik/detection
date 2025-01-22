from minio import Minio
import os

# 下载AI平台所需模型文件
# UI界面：https://wlnfs.weileit.com
# 缺乏依赖请安装依赖： pip install minio

# 创建一个Minio客户端
client = Minio(
    "wlnfsapi.weileit.com",
    access_key="admin",
    secret_key="Weileit_310",
    secure=True
)

bucket_name = "aimodels"
local_dir = "./default_models"

# 确保本地目录存在
os.makedirs(local_dir, exist_ok=True)

# 列出所有对象
objects = client.list_objects(bucket_name, recursive=True)

for obj in objects:
    # 构建本地文件路径
    local_file_path = os.path.join(local_dir, obj.object_name)
    local_file_dir = os.path.dirname(local_file_path)

    # 确保本地文件目录存在
    os.makedirs(local_file_dir, exist_ok=True)

    # 下载对象
    client.fget_object(bucket_name, obj.object_name, local_file_path)
    print(f"Downloaded {obj.object_name} to {local_file_path}")