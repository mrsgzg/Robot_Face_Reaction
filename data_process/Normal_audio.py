import os
import numpy as np
import joblib
import glob
import torchaudio
import torchaudio.transforms as T

# 1️⃣ 获取所有音频文件路径
all_files = glob.glob("Robot_dataset/Audio_files/*/*/*/*.wav")

# 2️⃣ 初始化 `StandardScaler`
from sklearn.preprocessing import StandardScaler
mfcc_scaler = StandardScaler()
mel_scaler = StandardScaler()

count = 0

# 3️⃣ **遍历所有音频，计算归一化参数**
for file in all_files:
    print(f"Processing: {file}")

    # 读取音频
    waveform, sr = torchaudio.load(file)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # 转换为单声道

    # **计算 hop_length 确保音频和视频对齐**
    fps = 25
    hop_length = int(sr / fps)
    target_frames = 750  

    # **计算 Mel 频谱**
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=40,n_fft=1024,hop_length=hop_length)
    mel_spec = mel_transform(waveform).squeeze(0).numpy()  # shape: (40, 时间帧数)

    # **计算 MFCC**
    mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={"hop_length": hop_length,"n_mels": 40,"n_fft": 1024,})
    mfcc = mfcc_transform(waveform).squeeze(0).numpy()  # shape: (13, 时间帧数)

    # **确保音频帧数对齐**
    current_frames = mfcc.shape[-1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="edge")
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="edge")
    elif current_frames > target_frames:
        mfcc = mfcc[:, :target_frames]
        mel_spec = mel_spec[:, :target_frames]

    # **计算 MFCC Δ 和 ΔΔ**
    mfcc_delta = np.gradient(mfcc, axis=1)
    mfcc_delta2 = np.gradient(mfcc_delta, axis=1)
    mfccs_all = np.stack((mfcc, mfcc_delta, mfcc_delta2), axis=0)  # shape: (3, 13, 750)
    mfcc_features = mfccs_all.transpose(2, 0, 1).reshape(target_frames, -1)  # 新 shape: (target_frames, 39)
    mel_features = mel_spec.T
    # **归一化**
    mfcc_scaler.partial_fit(mfcc_features)  
    mel_scaler.partial_fit(mel_features)  
    count += 1

# 4️⃣ 保存 `audio_scaler.pkl`
joblib.dump((mfcc_scaler, mel_scaler), "Audio_Scaler.pkl")

print(f"✅ 计算完成！共处理 {count} 个音频文件，归一化参数已保存为 `Audio_Scaler.pkl`")
