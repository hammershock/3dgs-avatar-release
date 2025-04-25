import cv2


def save_video_from_frames(frames, out_path, fps=50):
    if len(frames) == 0:
        print("Frame list is empty.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {out_path}")
