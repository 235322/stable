import numpy as np
import cv2
from config import GRID_SIZE, MAX_TRANSLATION_THRESHOLD, MAX_ROTATION_THRESHOLD

# 计算每个网格的变换轨迹
def compute_grid_trajectories(prev_gray, curr_gray, grid_size=GRID_SIZE):
    grid_w = prev_gray.shape[1] // grid_size[0]
    grid_h = prev_gray.shape[0] // grid_size[1]
    grid_trajectories = np.zeros(3, np.float32)  # 一维数组，存储 dx, dy, da

    grid_point = []
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            grid_x, grid_y = x * grid_w, y * grid_h
            grid_roi_prev = prev_gray[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]
            harris_response = cv2.cornerHarris(grid_roi_prev, 2, 3, 0.04)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris_response)
            grid_point.append((grid_x + max_loc[0], grid_y + max_loc[1]))

    good_point = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=30, blockSize=3)

    # 将 grid_point 和 good_point 分别转换为适合光流计算的格式
    prev_pts = np.array(grid_point, dtype=np.float32).reshape(-1, 1, 2)  # 网格角点
    good_pts = np.array(good_point, dtype=np.float32).reshape(-1, 1, 2)  # 全局角点

    # 正向光流跟踪
    grid_curr_pts, status_grid, err_grid = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_curr_pts, status_good, err_good = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, good_pts, None)

    # 反向光流跟踪
    grid_back_pts, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, grid_curr_pts, None)
    good_back_pts, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, good_curr_pts, None)

    # 重跟踪误差计算，同时筛选前一帧的点和后一帧的点
    reliable_prev_pts = []
    reliable_curr_pts = []

    for i, (p0, p1, p2) in enumerate(zip(prev_pts, grid_curr_pts, grid_back_pts)):
        if status_grid[i] and np.linalg.norm(p0 - p2) < 1.0:  # 判定误差阈值
            reliable_prev_pts.append(p0)
            reliable_curr_pts.append(p1)

    for i, (p0, p1, p2) in enumerate(zip(good_pts, good_curr_pts, good_back_pts)):
        if status_good[i] and np.linalg.norm(p0 - p2) < 1.0:
            reliable_prev_pts.append(p0)
            reliable_curr_pts.append(p1)

    # 转换为 numpy 格式，确保匹配的点数量一致
    reliable_prev_pts = np.array(reliable_prev_pts, dtype=np.float32).reshape(-1, 1, 2)
    reliable_curr_pts = np.array(reliable_curr_pts, dtype=np.float32).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffinePartial2D(reliable_prev_pts, reliable_curr_pts)
    if M is None:
        M = np.eye(2, 3, dtype=np.float32)

    dx = M[0, 2]
    dy = M[1, 2]
    da = np.arctan2(M[1, 0], M[0, 0])

    # 判断是否超出阈值
    if abs(dx) > MAX_TRANSLATION_THRESHOLD or abs(dy) > MAX_TRANSLATION_THRESHOLD or abs(da) > np.deg2rad(MAX_ROTATION_THRESHOLD):
        return None  # 若超出阈值则不计算

    grid_trajectories[0] = dx
    grid_trajectories[1] = dy
    grid_trajectories[2] = da
    return grid_trajectories
