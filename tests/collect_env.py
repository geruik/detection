# -*- encoding: utf-8 -*-
"""
AI平台-运行环境-检测-脚本.
"""
import importlib
import numpy as np
import os
import re
import subprocess
import sys
from collections import defaultdict
import PIL
import torch
import torchvision
from tabulate import tabulate

__all__ = ["collect_env_info"]


def getCudnnVersion(ver: int = 8):
    """获取系统安装的cudnn库的版本号

    Args:
        ver (int, optional): 大版本号. Defaults to 8.

    Returns:
        _type_: 版本号。"<not install>"代表未安装
    """
    import ctypes

    try:
        # 尝试加载libcudnn.so
        library = ctypes.LibraryLoader(ctypes.CDLL)
        libcudnn = library.LoadLibrary(f"libcudnn.so.{ver}")
        return libcudnn.cudnnGetVersion()
    except Exception as _:
        return "<not install>"


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line))
            arch = sorted(set(arch))
            return ", ".join(arch)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info():
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    # print system compilers when extension fails to build
    if sys.platform != "win32":  # don't know what to do for windows
        try:
            # this is how torch/utils/cpp_extensions.py choose compiler
            cxx = os.environ.get("CXX", "c++")
            cxx = subprocess.check_output("'{}' --version".format(cxx), shell=True)
            cxx = cxx.decode("utf-8").strip().split("\n")[0]
        except subprocess.SubprocessError:
            cxx = "Not found"
        data.append(("Compiler ($CXX)", cxx))

        if has_cuda and CUDA_HOME is not None:
            try:
                nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = subprocess.check_output("'{}' -V".format(nvcc), shell=True)
                nvcc = nvcc.decode("utf-8").strip().split("\n")[-1]
            except subprocess.SubprocessError:
                nvcc = "Not found"
            data.append(("CUDA compiler", nvcc))

    # data.append(get_env_module())
    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))
    try:
        import onnxruntime

        data.append(("ONNX Runtime:", onnxruntime.__version__))
        # 检查是否有CUDA加速
        available_providers = onnxruntime.get_available_providers()
        data.append(("| - available_providers", available_providers))
        # 检查当前设备是GPU还是CPU
        device = onnxruntime.get_device()
        data.append(("| - current device", device))
    except Exeception as e:
        pass

    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("GPU available", has_gpu))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            msg = " - invalid!" if not (ROCM_HOME and os.path.isdir(ROCM_HOME)) else ""
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))
        else:
            msg = " - invalid!" if not (CUDA_HOME and os.path.isdir(CUDA_HOME)) else ""
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    if sys.platform != "win32":
        data.append(("cuDNN 8 version", getCudnnVersion(8)))
        data.append(("cuDNN 9 version", getCudnnVersion(9)))
    data.append(("Pillow", PIL.__version__))

    try:
        data.append(
            (
                "torchvision",
                str(torchvision.__version__)
                + " @"
                + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        data.append(("cv2", "Not found"))
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":

    print(collect_env_info())

    # check all GPUs are working with cuda and torch
    if torch.cuda.is_available():
        for k in range(torch.cuda.device_count()):
            device = f"cuda:{k}"
            try:
                x = torch.tensor([1, 2.0], dtype=torch.float32)
                x = x.to(device)
            except Exception as e:
                print(
                    f"Unable to copy tensor to device={device}: {e}. "
                    "Your CUDA environment is broken."
                )
