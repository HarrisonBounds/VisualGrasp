import pyrealsense2 as rs
import numpy as np
import cv2

def capture_realsense_frame():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    
    for _ in range(10):  # let auto-exposure settle
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise RuntimeError("Could not retrieve frames from RealSense camera")

    color_image = np.asanyarray(color_frame.get_data())  # (480, 640, 3), uint8
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001  # in meters

    # Get intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam_K = np.array([
        [intr.fx,    0,      intr.ppx],
        [0,      intr.fy,    intr.ppy],
        [0,          0,          1   ]
    ], dtype=np.float32)

    pipeline.stop()

    return color_image, depth_image, cam_K

capture_realsense_frame()