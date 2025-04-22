import cv2
import numpy as np
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH
from smoothing import smooth_trajectory

def open_video(video_path=VIDEO_PATH):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()
    return cap

# 创建视频写入对象
def create_video_writer(width, height, fps, output_path=OUTPUT_VIDEO_PATH):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 获取视频属性
def get_video_properties(cap):
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return n_frames, width, height, fps

# 更新变换轨迹
def update_trajectories(grid_trajectories):
    # 计算累积变换轨迹
    trajectory = np.cumsum(grid_trajectories, axis=0)
    # 平滑变换轨迹
    smoothed_trajectory = smooth_trajectory(trajectory)
    # 计算平滑轨迹与原始轨迹的差异
    difference = smoothed_trajectory - trajectory
    # 更新变换数组
    grid_trajectories = grid_trajectories + difference
    return grid_trajectories



