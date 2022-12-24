from lib.config import cfg, args
import numpy as np
from lib.utils.pvnet import pvnet_config
from lib.utils.pvnet import pvnet_pose_utils
# from lib.csrc.uncertainty_pnp import un_pnp_utils
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
import cv2
import pyrealsense2 as rs
import json
import scipy.stats as ss
from scipy.io import savemat

# 切换不同的网络要改的地方：
# 1. train.json的路径
# 2. lib/config/config.py中_heads_factory特征点数量
# 3. custom.yaml中的模型
# 4. 可能要改相机的分辨率

# arguments: --type demo --cfg_file configs/custom.yaml
if __name__ == '__main__':
    # load the information of object from meta.npy
    # meta = np.load('test_images/meta.npy', allow_pickle=True).item()
    # kpt_3d = np.array(meta['kpt_3d'])
    # # K = np.array(meta['K'])  # do not use the saved camera intrinsic matrix
    # corner_3d = np.array(meta['corner_3d'])

    # load the object information from train.json
    f_json = open("data/custom/train.json")
    meta_json = json.load(f_json)
    fps_3d = meta_json['annotations'][0]['fps_3d']
    center_3d = meta_json['annotations'][0]['center_3d']
    corner_3d = np.asarray(meta_json['annotations'][0]['corner_3d'])
    kpt_3d = np.concatenate((fps_3d, [center_3d]), axis=0)
    f_json.close()
    # load network
    torch.manual_seed(0)
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()
    mean = pvnet_config.mean
    std = pvnet_config.std
    # configure realsense stream
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)  # 640 480
    pipeline.start(rs_cfg)
    # get the camera intrinsic matrix K
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    kpt_2d_list = []
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # color_image = color_image[960:1920, 540:1080, :]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
            img_ipnp = color_image.copy()
            # img_unpnp = color_image.copy()
            # pass through the network
            inp = (((color_image / 255.) - mean) / std).transpose(2, 0, 1).astype(np.float32)
            inp = torch.Tensor(inp[None]).cuda()
            with torch.no_grad():
                output = network(inp)
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            mask = output['mask'][0].detach().cpu().numpy().astype(np.uint8)
            kpt_idx = np.arange(0, kpt_2d.shape[0])

            # if cfg.test.un_pnp:
            #     var = output['var'][0].detach().cpu().numpy()
            #     # use the points with small covariance matrix traces
            #     trace_list = [np.matrix.trace(var[i, :, :]) for i in range(var.shape[0])]
            #     trace_rank = ss.rankdata(trace_list)
            #     kpt_idx = np.where(trace_rank <= 7)

            # PnP and plot the key points, bounding box
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            line_order = [0, 2, 6, 4, 0, 1, 5, 7, 3, 1, 3, 2, 6, 7, 5, 4]
            if kpt_2d.max() > 0:  # all zeros means detection failed
                kpt_2d_list.append(kpt_2d)
                pose_ipnp = pvnet_pose_utils.pnp(kpt_3d[kpt_idx, :], kpt_2d[kpt_idx, :], K)
                corner_2d_ipnp = pvnet_pose_utils.project(corner_3d, K, pose_ipnp).astype(int)
                # if cfg.test.un_pnp:
                #     pose_unpnp = un_pnp_utils.uncertainty_pnp_v2(kpt_2d, var, kpt_3d, K)
                #     corner_2d_unpnp = pvnet_pose_utils.project(corner_3d, K, pose_unpnp).astype(int)

                # debug: if the corner_2d_pred is problematic
                # dist_list = []
                # for i in range(8):
                #     dist_list.append(np.linalg.norm(corner_2d_pred[i, :] - corner_2d_pred[0, :]))
                # if np.max(dist_list) > 200:  # found the problematic frame and save the data
                #     debug = 1
                #     np.save("/home/wenbin/Desktop/kpt_2d.npy", kpt_2d)
                #     cv2.imwrite("/home/wenbin/Desktop/img.jpg", color_image)
                #     break

                # visualize the key-points and bounding box
                for point in kpt_2d:
                    img_ipnp = cv2.circle(color_image, tuple(point), 2, (0, 255, 0), -1)  # BGR
                    # img_unpnp = cv2.circle(color_image, tuple(point), 2, (0, 255, 0), -1)  # BGR
                for i in range(len(line_order) - 1):
                    img_ipnp = cv2.line(img_ipnp, tuple(corner_2d_ipnp[line_order[i], :]),
                             tuple(corner_2d_ipnp[line_order[i+1], :]), (0, 0, 0), 2)
                    # if cfg.test.un_pnp:
                    #     img_ipnp = cv2.line(img_unpnp, tuple(corner_2d_unpnp[line_order[i], :]),
                    #                         tuple(corner_2d_unpnp[line_order[i + 1], :]), (0, 0, 0), 2)

            # plot the rgb image
            cv2.namedWindow('Iterative PnP', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Iterative PnP', cv2.cvtColor(img_ipnp, cv2.COLOR_RGB2BGR))
            # if cfg.test.un_pnp:
            #     cv2.namedWindow('Uncertainty PnP', cv2.WINDOW_AUTOSIZE)
            #     cv2.imshow('Uncertainty PnP', cv2.cvtColor(img_unpnp, cv2.COLOR_RGB2BGR))

            # plot the mask
            cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Mask', cv2.bitwise_and(img_ipnp, img_ipnp, mask=mask))

            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()
        # savemat('tmp.mat', {'kpts': kpt_2d_list})