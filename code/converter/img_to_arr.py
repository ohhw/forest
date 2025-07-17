from PIL import Image
from numpy import asarray
import numpy as np
import os

img_path = r"C:/Users/USER/Desktop/data/"
file_name = 'Camera1_1_crop_bmp.rf.a7d83016f6268bbd3bf9ed0d09700e6c_mask.png'

full_path = os.path.join(img_path, file_name)
img = Image.open(full_path)

numpydata = asarray(img)

output_file = r'C:\Users\USER\Desktop\output_array.txt'

# Numpy 배열을 텍스트 파일로 저장
if numpydata.ndim == 3:
    # 다채널 이미지일 경우 (예: RGB), 각 픽셀 값은 쉼표로 구분
    np.savetxt(output_file, numpydata.reshape(-1, numpydata.shape[-1]), fmt='%d', delimiter=',')
else:
    # 단일 채널 이미지일 경우 (예: Grayscale)
    np.savetxt(output_file, numpydata, fmt='%d')