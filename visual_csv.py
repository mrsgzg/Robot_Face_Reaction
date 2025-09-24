import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_openface_csv(file_path):
    df = pd.read_parquet(file_path)
    

    landmark_x_cols = [f"x_{i}" for i in range(68)]
    landmark_y_cols = [f"y_{i}" for i in range(68)]
    
    if not set(landmark_x_cols).issubset(df.columns) or not set(landmark_y_cols).issubset(df.columns):
        raise ValueError("CSV error！")
    
    landmarks_x = df[landmark_x_cols].values
    landmarks_y = df[landmark_y_cols].values
    
    min_x, max_x = landmarks_x.min(), landmarks_x.max()
    min_y, max_y = landmarks_y.min(), landmarks_y.max()
    
    landmarks_x = (landmarks_x - min_x) / (max_x - min_x)
    landmarks_y = (landmarks_y - min_y) / (max_y - min_y)
    
    if {'gaze_angle_x', 'gaze_angle_y'}.issubset(df.columns):
        gaze_x = df['gaze_angle_x'].values
        gaze_y = df['gaze_angle_y'].values
    else:
        gaze_x, gaze_y = None, None
    
    return landmarks_x, landmarks_y, gaze_x, gaze_y

def animate_face(csv_path):
    landmarks_x, landmarks_y, gaze_x, gaze_y = load_openface_csv(csv_path)
    num_frames = landmarks_x.shape[0]
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  
    ax.set_title("OpenFace 2D Facial Landmarks & Gaze Direction Animation")
    
    scatter, = ax.plot([], [], 'ro', markersize=2)
    gaze_line_left, = ax.plot([], [], 'b-', linewidth=2)  
    gaze_line_right, = ax.plot([], [], 'b-', linewidth=2) 
    
    def update(frame):
        scatter.set_data(landmarks_x[frame], landmarks_y[frame])
        
        if gaze_x is not None and gaze_y is not None:
            
            left_eye_x = (landmarks_x[frame][36] + landmarks_x[frame][39]) / 2
            left_eye_y = (landmarks_y[frame][36] + landmarks_y[frame][39]) / 2
            left_gaze_end_x = left_eye_x + 0.1 * gaze_x[frame]
            left_gaze_end_y = left_eye_y - 0.1 * gaze_y[frame]
            gaze_line_left.set_data([left_eye_x, left_gaze_end_x], [left_eye_y, left_gaze_end_y])
            
           
            right_eye_x = (landmarks_x[frame][42] + landmarks_x[frame][45]) / 2
            right_eye_y = (landmarks_y[frame][42] + landmarks_y[frame][45]) / 2
            right_gaze_end_x = right_eye_x + 0.1 * gaze_x[frame]
            right_gaze_end_y = right_eye_y - 0.1 * gaze_y[frame]
            gaze_line_right.set_data([right_eye_x, right_gaze_end_x], [right_eye_y, right_gaze_end_y])
        
        return scatter, gaze_line_left, gaze_line_right
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)
    ani.save("test.gif", writer='pillow')
    plt.close(fig)
    

csv_file_path = "Robot_dataset/Face/NoXI/072_2016-05-23_Augsburg/Expert_video/1.parquet"  
animate_face(csv_file_path)
