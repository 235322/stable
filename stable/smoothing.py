import numpy as np
from scipy.signal import savgol_filter
from config import SMOOTHING_RADIUS

# 移动平均平滑
def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

# 平滑轨迹
def smooth_trajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
        smoothed_trajectory[:, i] = savgol_filter(smoothed_trajectory[:, i], window_length=51, polyorder=3)
    return smoothed_trajectory
