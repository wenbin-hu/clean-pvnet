from lib.config import cfg, args
import numpy as np
from lib.utils.pvnet import pvnet_config
from lib.utils.pvnet import pvnet_pose_utils
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
import cv2
import pyrealsense2 as rs

if __name__ == '__main__':
    torch.manual_seed(0)
    # load the information of object
    meta = np.load('test_images/meta.npy', allow_pickle=True).item()
    kpt_3d = np.array(meta['kpt_3d'])
    # K = np.array(meta['K'])  # do not use the saved camera intrinsic matrix
    corner_3d = np.array(meta['corner_3d'])
    # load network
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()
    mean = pvnet_config.mean
    std = pvnet_config.std
    # configure realsense stream
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_cfg)
    # get the camera intrinsic matrix K
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # pass through the network
            inp = (((color_image / 255.) - mean) / std).transpose(2, 0, 1).astype(np.float32)
            inp = torch.Tensor(inp[None]).cuda()
            with torch.no_grad():
                output = network(inp)
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            # PnP and plot the key points, bounding box
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            line_order = [0, 2, 6, 4, 0, 1, 5, 7, 3, 1, 3, 2, 6, 7, 5, 4]
            if kpt_2d.max() > 0:  # all zeros means detection failed
                pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
                corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred).astype(int)
                for point in kpt_2d:
                    cv2.circle(color_image, tuple(point), 2, (0, 255, 0), -1)  # BGR
                # for i in range(8):
                #     cv2.circle(color_image, tuple(corner_2d_pred[i, :]), 5, color[i], -1)
                for i in range(len(line_order) - 1):
                    cv2.line(color_image, tuple(corner_2d_pred[line_order[i], :]),
                             tuple(corner_2d_pred[line_order[i+1], :]), (0, 0, 0), 2)
            # plot the rgb image
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()