import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.preprocessing import StandardScaler

# 1️⃣ 获取所有 parquet 文件路径
all_files = glob.glob("Robot_dataset/Face/*/*/*/*.parquet")
total_files = len(all_files)  # 获取文件总数

# 2️⃣ 初始化 StandardScaler
scaler = StandardScaler()
count = 0

# 3️⃣ 遍历文件，增量更新 `mean` 和 `std`
for i, file in enumerate(all_files):
    print(f"正在处理文件 {i+1}/{total_files}: {file}")  # 输出当前文件处理进度
    try:
        # 读取 parquet 文件
        df = pd.read_parquet(file)
        df.columns = df.columns.str.replace(" ", "")  # 去除列名中的空格

        # 选择需要的特征
        features = [f"x_{i}" for i in range(68)] + \
                   [f"y_{i}" for i in range(68)] + \
                   [f"AU{i:02d}_r" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]] + \
                   ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"] + \
                   ["gaze_angle_x", "gaze_angle_y"]

        # 确保所有特征都存在
        if all(feature in df.columns for feature in features):
            df = df.astype('float32')
            data = df[features].to_numpy()
            scaler.partial_fit(data)  # 增量更新归一化参数
            count += data.shape[0]
        else:
            missing_features = [feature for feature in features if feature not in df.columns]
            print(f"警告: 文件 {file} 缺失特征 {missing_features}")

    except Exception as e:
        print(f"处理文件 {file} 时出现错误: {e}")

    # 每处理 100 个文件输出一次进度
    if (i + 1) % 100 == 0 or (i + 1) == total_files:
        print(f"已处理 {i+1}/{total_files} 个文件, 当前已处理 {count} 行数据")

# 4️⃣ 保存 scaler 参数
joblib.dump(scaler, "Face_Scaler.pkl")

print(f"✅ 计算完成！共处理 {count} 行数据，归一化参数已保存为 `Face_scaler.pkl`")
