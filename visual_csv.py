import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 读取 OpenFace 生成的 CSV 文件
def load_openface_csv(file_path):
    df = pd.read_parquet(file_path)
    
    # 去除列名的前后空格
    #df.columns = df.columns.str.strip()
    
    # 提取 68 个特征点的 x, y 坐标
    landmark_x_cols = [f"x_{i}" for i in range(68)]
    landmark_y_cols = [f"y_{i}" for i in range(68)]
    
    if not set(landmark_x_cols).issubset(df.columns) or not set(landmark_y_cols).issubset(df.columns):
        raise ValueError("CSV 文件中没有找到完整的 68 个人脸特征点坐标！")
    
    landmarks_x = df[landmark_x_cols].values
    landmarks_y = df[landmark_y_cols].values
    
    # 计算全局最小值和最大值进行归一化
    min_x, max_x = landmarks_x.min(), landmarks_x.max()
    min_y, max_y = landmarks_y.min(), landmarks_y.max()
    
    landmarks_x = (landmarks_x - min_x) / (max_x - min_x)
    landmarks_y = (landmarks_y - min_y) / (max_y - min_y)
    
    # 提取 Gaze 方向信息
    if {'gaze_angle_x', 'gaze_angle_y'}.issubset(df.columns):
        gaze_x = df['gaze_angle_x'].values
        gaze_y = df['gaze_angle_y'].values
    else:
        gaze_x, gaze_y = None, None
    
    return landmarks_x, landmarks_y, gaze_x, gaze_y

# 2D 动画绘制人脸特征点和眼神方向
def animate_face(csv_path):
    landmarks_x, landmarks_y, gaze_x, gaze_y = load_openface_csv(csv_path)
    num_frames = landmarks_x.shape[0]
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # 反转 y 轴使得面部显示正常
    ax.set_title("OpenFace 2D Facial Landmarks & Gaze Direction Animation")
    
    scatter, = ax.plot([], [], 'ro', markersize=2)
    gaze_line_left, = ax.plot([], [], 'b-', linewidth=2)  # 左眼视线
    gaze_line_right, = ax.plot([], [], 'b-', linewidth=2)  # 右眼视线
    
    def update(frame):
        scatter.set_data(landmarks_x[frame], landmarks_y[frame])
        
        if gaze_x is not None and gaze_y is not None:
            # 左眼中心 (Eye landmark 36, 39)
            left_eye_x = (landmarks_x[frame][36] + landmarks_x[frame][39]) / 2
            left_eye_y = (landmarks_y[frame][36] + landmarks_y[frame][39]) / 2
            left_gaze_end_x = left_eye_x + 0.1 * gaze_x[frame]
            left_gaze_end_y = left_eye_y - 0.1 * gaze_y[frame]
            gaze_line_left.set_data([left_eye_x, left_gaze_end_x], [left_eye_y, left_gaze_end_y])
            
            # 右眼中心 (Eye landmark 42, 45)
            right_eye_x = (landmarks_x[frame][42] + landmarks_x[frame][45]) / 2
            right_eye_y = (landmarks_y[frame][42] + landmarks_y[frame][45]) / 2
            right_gaze_end_x = right_eye_x + 0.1 * gaze_x[frame]
            right_gaze_end_y = right_eye_y - 0.1 * gaze_y[frame]
            gaze_line_right.set_data([right_eye_x, right_gaze_end_x], [right_eye_y, right_gaze_end_y])
        
        return scatter, gaze_line_left, gaze_line_right
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)
    ani.save("test.gif", writer='pillow')
    plt.close(fig)
    

# 使用示例
csv_file_path = "Robot_dataset/Face/NoXI/072_2016-05-23_Augsburg/Expert_video/1.parquet"  # 替换为你的 OpenFace CSV 文件路径
animate_face(csv_file_path)
