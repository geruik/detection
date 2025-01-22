# -*- encoding: utf-8 -*-
"""
用于打印内存中对象计数和大小的调试脚本.
"""
import sys
import gc

# 首先垃圾回收
gc.collect()

# 获取当前内存中的所有对象
objects = gc.get_objects()

# 初始化计数器
object_count = {}
object_size = {}
total_size = 0
"""内存总占用大小"""

# 遍历所有对象并统计计数和大小
for obj in objects:
    obj_type = type(obj)
    object_count[obj_type] = object_count.get(obj_type, 0) + 1
    obj_size = sys.getsizeof(obj)
    total_size += obj_size
    object_size[obj_type] = object_size.get(obj_type, 0) + obj_size

# 按照对象大小排序
sorted_objects = sorted(object_size.items(), key=lambda x: x[1], reverse=True)

print(f"所有对象占用内存大小：{total_size/1024/1024}M bytes")
print("对象占用内存情况:")
print("----")
# 打印对象计数和大小
for obj_type, size in sorted_objects:
    print(
        f"Type: {obj_type}, Count: {object_count[obj_type]}, Total Size: {size/1024}K bytes"
    )
