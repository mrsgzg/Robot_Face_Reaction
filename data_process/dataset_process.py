import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

class SpeakerListenerDataset(Dataset):
    def __init__(self, mapping_csv, stride=10, sequence_length=100, normalize=True, scaler_path="scaler.pkl", audio_scaler_path="audio_scaler.pkl"):
        
        self.target_length = 750  # ✅ 统一长度为 750 帧
        self.mapping_df = pd.read_csv(mapping_csv, engine="c")  # ✅ 使用 C 解析器加速
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        # **🔹 加载全局归一化参数**

        self.scaler = joblib.load(scaler_path)  
        # **加载音频 scaler**
        self.mfcc_scaler, self.mel_scaler = joblib.load(audio_scaler_path)  


    def _convert_path(self, path, ext):
        """自动适配 Windows 和 Ubuntu 路径，同时替换扩展名"""
        path = "Robot_dataset/Face/"+path+"."+ext
        return path
    
    def _convert_audio_path(self, path,ext):
        path = "Robot_dataset/Audio_files/"+path+"."+ext
        return path
    def _interpolate_missing_frames(self, df):
        """填充缺失帧，确保 750 帧"""
        if "frame" not in df.columns:
            return df  # 若无 frame 列，则直接返回
        if len(df) == self.target_length:
            return df

        full_frame_range = np.arange(0, self.target_length)  # 生成 0~749 帧
        df = df.set_index("frame").reindex(full_frame_range)  # 重新索引
        df.interpolate(method="linear", inplace=True)  # 线性插值
        df.ffill(inplace=True)  # 向前填充
        df.bfill(inplace=True)  # 向后填充

        return df.reset_index()

    def _load_parquet(self, parquet_path):
        """读取 Parquet 并进行特征筛选 + 归一化 + 插值补帧"""
        if not os.path.exists(parquet_path) or os.stat(parquet_path).st_size == 0:
            print(f"❌ 文件不存在或为空: {parquet_path}")
            return None

        try:
            df = pd.read_parquet(parquet_path)  # 读取 Parquet
            
            df = df.loc[:self.target_length - 1]  # ✅ 限制为 750 帧
            df = self._interpolate_missing_frames(df)  # **✅ 插值补帧**
            df = df.drop(columns=["frame"])  # 删除 frame 列多余列
            print("✅ 加载成功:", parquet_path)
            # **使用全局 scaler 归一化**    
            df = self.scaler.transform(df.to_numpy())  # ✅ 使用全局 scaler 归一化

            return df
        except Exception as e:
            print(f"⚠️  加载失败: {parquet_path} - {e}")
            return None


    def _load_audio(self, audio_path):
        """使用 torchaudio 读取 WAV 音频并提取 MFCC + Δ + Mel 频谱特征"""
        if not os.path.exists(audio_path):
            print(f"❌ 音频文件不存在: {audio_path}")
            return None, None

        try:
            # 1️⃣ **读取音频**
            waveform, sr = torchaudio.load(audio_path)  # 读取音频
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # 转换为单声道

            # 2️⃣ **计算 hop_length 确保音频和视频帧率对齐**
            fps = 25  # 视频帧率 25 FPS
            hop_length = int(sr / fps)  # 计算 hop_length
            target_frames = 750  # 确保特征与视频对齐

            # 3️⃣ **计算 Mel 频谱**
            mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=40, n_fft=1024, hop_length=hop_length)
            mel_spec = mel_transform(waveform).squeeze(0).numpy()  # shape: (40, 时间帧数)
            #mel_spec = np.log1p(mel_spec)
            # 4️⃣ **计算 MFCC**
            mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={"n_mels": 40, "n_fft": 1024, "hop_length": hop_length})
            mfcc = mfcc_transform(waveform).squeeze(0).numpy()  # shape: (13, 时间帧数)

            # 5️⃣ **计算 MFCC Δ（速度）和 ΔΔ（加速度）**
            mfcc_delta = np.gradient(mfcc, axis=1)
            mfcc_delta2 = np.gradient(mfcc_delta, axis=1)
            mfccs_all = np.stack((mfcc, mfcc_delta, mfcc_delta2), axis=0)  # shape: (3, 13, 时间帧数)

            # 6️⃣ **确保音频帧数 >= 750**
            current_frames = mfccs_all.shape[-1]
            if current_frames < target_frames:
                pad_width = target_frames - current_frames
                mfccs_all = np.pad(mfccs_all, ((0, 0), (0, 0), (0, pad_width)), mode="edge")
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="edge")
            elif current_frames > target_frames:
                mfccs_all = mfccs_all[:, :, :target_frames]  # 截断
                mel_spec = mel_spec[:, :target_frames]  # 截断

            # 7️⃣ **归一化**
            mfccs_all = mfccs_all.transpose(2, 0, 1).reshape(target_frames, -1)
            mfccs_all = self.mfcc_scaler.transform(mfccs_all)
            
            mel_spec = mel_spec.T
            mel_spec = self.mel_scaler.transform(mel_spec)
            
            return mfccs_all, mel_spec  # 归一化后的 MFCC 和 Mel 频谱
        except Exception as e:
            print(f"⚠️  音频加载失败: {audio_path} - {e}")
            return None, None


    def _create_sequences(self, expr_array, mfcc, mel):
        
        num_sequences = (expr_array.shape[0] - self.sequence_length) // self.stride + 1
        
        expr_sequences = np.zeros((num_sequences, self.sequence_length, expr_array.shape[1]), dtype=np.float32)
        mfcc_sequences = np.zeros((num_sequences, self.sequence_length, mfcc.shape[1]), dtype=np.float32)
        mel_sequences = np.zeros((num_sequences, self.sequence_length, mel.shape[1]), dtype=np.float32)

        for i in range(num_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length

            expr_sequences[i] = expr_array[start_idx:end_idx]  # (sequence_length, 161)
            mfcc_sequences[i] = mfcc[start_idx:end_idx]  # (sequence_length, 39)
            mel_sequences[i] = mel[start_idx:end_idx]  # (sequence_length, 40)

        return expr_sequences, mfcc_sequences, mel_sequences


    def __len__(self):
        """返回数据集大小"""
        return len(self.mapping_df)

    def __getitem__(self, idx):
        """按索引读取数据（Lazy Loading）"""
        row = self.mapping_df.iloc[idx]
        speaker_parquet = self._convert_path(row.iloc[1], "parquet")
        listener_parquet = self._convert_path(row.iloc[2], "parquet")
        speaker_audio = self._convert_audio_path(row.iloc[1], "wav")
        listener_audio = self._convert_audio_path(row.iloc[2], "wav")

        speaker_df = self._load_parquet(speaker_parquet)
        listener_df = self._load_parquet(listener_parquet)
        speaker_mfcc, speaker_mel = self._load_audio(speaker_audio)
        listener_mfcc, listener_mel = self._load_audio(listener_audio)

        if any(x is None for x in [speaker_df, listener_df, speaker_mfcc, listener_mel, listener_mfcc, listener_mel]):
            print(f"⚠️  跳过索引 {idx}: {speaker_parquet} 或 {listener_parquet} 加载失败")
            return None

        speaker_sequences_expr, speaker_sequences_mfcc, speaker_sequences_mel = self._create_sequences(speaker_df, speaker_mfcc, speaker_mel)
        
        listener_sequences_expr, listener_sequences_mfcc, listener_sequences_mel = self._create_sequences(listener_df, listener_mfcc, listener_mel)

        return speaker_sequences_expr, listener_sequences_expr, speaker_sequences_mfcc, listener_sequences_mfcc, speaker_sequences_mel, listener_sequences_mel


def get_dataloader(mapping_csv, batch_size=2, stride=10, num_workers=0, sequence_length=100, scaler_path="scaler.pkl", audio_scaler_path="audio_scaler.pkl"):
    dataset = SpeakerListenerDataset(mapping_csv, stride, sequence_length, scaler_path=scaler_path, audio_scaler_path=audio_scaler_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    # ✅ 指定测试路径
    mapping_csv = "Robot_dataset/train.csv"
    face_scaler_path = "Robot_face_gen/data_process/Face_Scaler.pkl"
    audio_scaler_path = "Robot_face_gen/data_process/Audio_Scaler.pkl"
    # ✅ 创建 DataLoader
    dataloader = get_dataloader(mapping_csv, batch_size=2, stride=10, num_workers=0, sequence_length=100, scaler_path=face_scaler_path,audio_scaler_path = audio_scaler_path)

    # ✅ 取出一个 batch 并检查形状
    for i, (speaker_seq, listener_seq, speaker_mfcc, listener_mfcc, speaker_mel, listener_mel) in enumerate(dataloader):
        print(f"🟢 Batch {i} Loaded Successfully!")

        # **检查表情数据 Shape**
        print(f"🗿 Speaker Facial Expression Shape: {speaker_seq.shape}")  
        print(f"🗿 Listener Facial Expression Shape: {listener_seq.shape}")

        # **检查音频 MFCC + Δ + ΔΔ 特征 Shape**
        print(f"🎵 Speaker MFCC Shape: {speaker_mfcc.shape}")  
        print(f"🎵 Listener MFCC Shape: {listener_mfcc.shape}")

        # **检查 Mel 频谱 Shape**
        print(f"🎵 Speaker Mel Spectrogram Shape: {speaker_mel.shape}")  
        print(f"🎵 Listener Mel Spectrogram Shape: {listener_mel.shape}")

        # **只测试一个 batch**
        break
