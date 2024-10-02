import cv2
import numpy as np
import os
from scipy.io.wavfile import write

# 设置工作文件夹
path = 'C:/Users/Billy/Desktop/aa'
video_path = os.path.join(path, 'badApple.mp4')
bmp_folder = os.path.join(path, 'bmp')

# 创建存放 BMP 文件的文件夹
os.makedirs(bmp_folder, exist_ok=True)

# 读取原视频
raw_video = cv2.VideoCapture(video_path)
fps = raw_video.get(cv2.CAP_PROP_FPS)
num_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

# 设置高度并计算宽度以保持宽高比
height = 256
width = round(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH) / raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT) * height)

# 创建未压缩的 AVI 视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(os.path.join(path, 'badApple_Compressed.avi'), fourcc, fps, (width, height))

# 遍历视频每一帧
for i in range(num_frames):
    ret, frame = raw_video.read()
    if not ret:
        break
    new_frame = cv2.resize(frame, (width, height))
    video_writer.write(new_frame)
    cv2.imwrite(os.path.join(bmp_folder, f'{i + 1}.bmp'), new_frame)

raw_video.release()
video_writer.release()

# 找出轮廓并记录
edge_array = []
for i in range(num_frames):
    gray = cv2.cvtColor(cv2.imread(os.path.join(bmp_folder, f'{i + 1}.bmp')), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    y, x = np.where(edges == 255)
    edge_array.append(np.column_stack((x, height - y)))

# 按轮廓排序
for i in range(num_frames):
    len_edges = edge_array[i].shape[0]
    if len_edges > 2:
        edge_array[i] = edge_array[i][np.argsort(edge_array[i][:, 1])]

# 设置 WAV 文件头和数据
sampling_rate = 88200
sound_channel = 2
bits_per_sample = 8
samp_per_frame = round(sampling_rate / fps)
data_size = num_frames * samp_per_frame * sound_channel
file_size = data_size + 36
byte_rate = sampling_rate * sound_channel * bits_per_sample // 8
block_align = sound_channel * bits_per_sample // 8

# WAV 头
wav_header = bytearray()
wav_header += b'RIFF'
wav_header += (file_size).to_bytes(4, 'little')
wav_header += b'WAVEfmt '
wav_header += (16).to_bytes(4, 'little')
wav_header += (1).to_bytes(2, 'little')  # PCM
wav_header += (sound_channel).to_bytes(2, 'little')
wav_header += (sampling_rate).to_bytes(4, 'little')
wav_header += (byte_rate).to_bytes(4, 'little')
wav_header += (block_align).to_bytes(2, 'little')
wav_header += (bits_per_sample).to_bytes(2, 'little')
wav_header += b'data'
wav_header += (data_size).to_bytes(4, 'little')

# 数据部分
data = np.zeros(44 + data_size, dtype=np.uint8)  # 确保包含 WAV 头的大小
for i in range(num_frames):
    len_edges = edge_array[i].shape[0]
    if len_edges > 0:
        for j in range(samp_per_frame):
            # 这里使用线性插值或最近的有效点
            idx = min(int((j * len_edges) / samp_per_frame), len_edges - 1)
            # 计算当前采样在 data 数组中的位置
            data_index = 44 + (i * sound_channel * samp_per_frame) + j * 2
            if data_index + 1 < len(data):  # 确保索引不会超出 data 数组范围
                data[data_index] = edge_array[i][idx, 0]  # 左声道
                data[data_index + 1] = edge_array[i][idx, 1]  # 右声道


# 写入 WAV 文件
with open(os.path.join(path, 'badApple.wav'), 'wb') as wav_file:
    wav_file.write(wav_header)
    wav_file.write(data.tobytes())
