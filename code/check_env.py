import torch
import torchvision
import cv2
import numpy as np
import ultralytics
import platform
import sys
import os

print("✅ Python Version:", sys.version)
print("✅ OS:", platform.platform())

print("\n🧠 PyTorch Version:", torch.__version__)
print("🖼️  Torchvision Version:", torchvision.__version__)
print("💻  CUDA Available:", torch.cuda.is_available())
print("🎯  CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("📦  CUDA Device Name:", torch.cuda.get_device_name(0))
    print("🚀  Current CUDA Device ID:", torch.cuda.current_device())

print("\n🧪 Ultralytics YOLO Version:", ultralytics.__version__)
print("📦 OpenCV Version:", cv2.__version__)
print("🔢 NumPy Version:", np.__version__)

# ONNXRuntime은 선택적으로 출력
try:
    import onnxruntime
    print("🧩 ONNXRuntime Version:", onnxruntime.__version__)
    print("🧠 ONNXRuntime Providers:", onnxruntime.get_available_providers())
except ImportError:
    print("🧩 ONNXRuntime: Not installed (✔️ expected if not used)")

# 환경변수 확인
print("\n🌍 CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
print("🌍 LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "(not set)"))