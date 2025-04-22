import cv2
import numpy as np
from fixborder import fix_border
from trajectory import compute_grid_trajectories
from utils import open_video, create_video_writer, get_video_properties, update_trajectories
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH, GRID_SIZE


def main():
    # 打开视频文件
    cap = open_video(VIDEO_PATH)
    n_frames, width, height, fps = get_video_properties(cap)

    # 创建视频写入对象
    out = create_video_writer(width, height, fps)

    # 读取视频的第一帧
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    grid_trajectories = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        grid_trajectories_i = compute_grid_trajectories(prev_gray, curr_gray)
        if grid_trajectories_i is None:
            print(f"Skipping frame {i+1} due to excessive motion")
            continue  # 如果变换量超出阈值，跳过当前帧的消抖处理

        grid_trajectories[i, :] = grid_trajectories_i

        prev_gray = curr_gray

    # 计算累积变换轨迹并平滑
    grid_trajectories = update_trajectories(grid_trajectories)

    # 重置视频读取位置到第一帧并进行稳定化
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames - 2):
        success, frame = cap.read()
        if not success:
            break

        # 获取平滑变换参数
        dx = grid_trajectories[i, 0]
        dy = grid_trajectories[i, 1]
        da = grid_trajectories[i, 2]

        # 构造仿射变换矩阵
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # 应用变换并修复边界问题
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        frame_stabilized = fix_border(frame_stabilized)

        # 将原始帧与平滑帧并排放置
        frame_out = cv2.hconcat([frame, frame_stabilized])
        out.write(frame_stabilized)

if __name__ == "__main__":
    main()


