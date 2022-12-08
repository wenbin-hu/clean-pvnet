import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from lib.utils.pvnet import pvnet_pose_utils
import json
import torch
from lib.utils.net_utils import load_network
from lib.config import cfg, args
from lib.networks import make_network
from lib.utils.pvnet import pvnet_config
from lib.utils.pvnet import pvnet_pose_utils

if __name__ == '__main__':
    # load data
    f_json = open("data/custom/train.json")
    meta_json = json.load(f_json)
    fps_3d = meta_json['annotations'][0]['fps_3d']
    center_3d = meta_json['annotations'][0]['center_3d']
    corner_3d = np.asarray(meta_json['annotations'][0]['corner_3d'])
    kpt_3d = np.concatenate((fps_3d, [center_3d]), axis=0)
    f_json.close()
    K = np.loadtxt("/home/wenbin/clean-pvnet/data/custom/camera.txt")
    bgr = cv2.imread("/home/wenbin/Desktop/img.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # load network
    torch.manual_seed(0)
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()
    mean = pvnet_config.mean
    std = pvnet_config.std

    # pass through the network
    inp = (((rgb / 255.) - mean) / std).transpose(2, 0, 1).astype(np.float32)
    inp = torch.Tensor(inp[None]).cuda()
    with torch.no_grad():
        output = network(inp)
    kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
    mask = output['mask'][0].detach().cpu().numpy()
    pose = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
    corner_2d = pvnet_pose_utils.project(corner_3d, K, pose).astype(int)
    # masked object
    mug_idxs = np.where(mask == 1)
    mask_rgb = np.zeros(rgb.shape)
    mask_rgb[mug_idxs[0], mug_idxs[1], :] = rgb[mug_idxs[0], mug_idxs[1], :]

    # plot
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(bgr)
    axs[0].plot(kpt_2d[:, 0], kpt_2d[:, 1], 'go', linewidth=2, markersize=3)
    axs[0].add_patch(patches.Polygon(xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
    axs[0].add_patch(patches.Polygon(xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
    axs[1].imshow(mask_rgb)
    plt.show()