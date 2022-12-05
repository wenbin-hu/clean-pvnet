import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
from handle_custom_dataset import get_model_corners
from tqdm import tqdm

# we assume the rgb and mask have the same size
def shift(rgb, mask, K, fps_2d, fps_3d, ud, vd, resize_rate=1.0):
    new_rgb = np.zeros(rgb.shape, dtype=np.uint8)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    new_fps_2d = fps_2d.copy()

    # resize
    h, w, _ = rgb.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, resize_rate)
    resized_rgb = cv2.warpAffine(rgb, M, (w, h))
    resized_mask = cv2.warpAffine(mask, M, (w, h))
    new_fps_2d = rotate_point(new_fps_2d, M)

    # shift
    for v in range(h):
        for u in range(w):
            if 0 <= v + vd < h and 0 <= u + ud < w:
                new_rgb[v+vd, u+ud, :] = resized_rgb[v, u, :]
                new_mask[v+vd, u+ud] = resized_mask[v, u]
    new_fps_2d = np.clip(new_fps_2d + [ud, vd], a_min=[0, 0], a_max=[w, h])  # BE careful of the order of ud vd

    # get the new camera pose;
    # in 'pvnet.py' the object center point positions are used as well. Here we only use surface key points.
    new_pose = pvnet_pose_utils.pnp(fps_3d, new_fps_2d, K)
    new_corner_2d = pvnet_pose_utils.project(corner_3d, K, new_pose)

    return new_rgb, new_mask, new_fps_2d, new_pose, new_corner_2d


def rotate_point(ps, m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = np.array([[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))])
    return target_point


def rotate(rgb, mask, K, fps_2d, fps_3d, angle, resize_rate=1.0):
    h, w, _ = rgb.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, resize_rate)
    new_rgb = cv2.warpAffine(rgb, M, (w, h))
    new_mask = cv2.warpAffine(mask, M, (w, h))
    new_fps_2d = rotate_point(fps_2d, M)

    new_pose = pvnet_pose_utils.pnp(fps_3d, new_fps_2d, K)
    new_corner_2d = pvnet_pose_utils.project(corner_3d, K, new_pose)

    return new_rgb, new_mask, new_fps_2d, new_pose, new_corner_2d


# input: original rgb, mask, pose, fps_3d, K
# output: modified rgb, mask, pose
# method:
# 1. pvnet_pose_utils.project(fps_3d, K, pose) --> fps_2d
# 2. modify the picture and get the new fps_2d
# 3. pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K) --> new pose
# 4. save the new rgb, mask, pose
if __name__ == '__main__':
    # path
    data_root = "/home/wenbin/clean-pvnet/data/original_mug/"
    pose_dir = os.path.join(data_root, 'pose')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')
    model_dir = os.path.join(data_root, 'model.ply')
    output_root = "/home/wenbin/clean-pvnet/data/custom/"
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(output_root + "rgb/"):
        os.makedirs(output_root + "rgb/")
    if not os.path.exists(output_root + "mask/"):
        os.makedirs(output_root + "mask/")
    if not os.path.exists(output_root + "pose/"):
        os.makedirs(output_root + "pose/")

    # load the original data
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))
    renderer = OpenGLRenderer(model_dir)
    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)

    pbar = tqdm(total=600)
    i = 0
    while i < 600:
        rgb = cv2.imread(rgb_dir + '/%d.jpg'%i)
        mask = cv2.imread(mask_dir + '/%d.png'%i)
        pose = np.load(pose_dir + '/%d.npy'%i, allow_pickle=True)
        corner_2d = pvnet_pose_utils.project(corner_3d, K, pose)

        angle = np.random.randint(low=-60, high=60)
        scale = np.random.uniform(low=0.5, high=1.5)
        dx = np.random.randint(low=-200, high=200)
        dy = np.random.randint(low=-200, high=200)
        # first scale and rotate
        fps_2d = pvnet_pose_utils.project(fps_3d, K, pose)
        new_rgb, new_mask, new_fps_2d, _, _ = rotate(rgb=rgb, mask=mask, K=K,
                                                        fps_2d=fps_2d, fps_3d=fps_3d, angle=angle, resize_rate=scale)
        # then shift
        new_rgb, new_mask, new_fps_2d, new_pose, new_corner_2d = shift(rgb=new_rgb, mask=new_mask, K=K,
                                                   fps_2d=new_fps_2d, fps_3d=fps_3d, ud=dx, vd=dy, resize_rate=1.0)
        # if the object is out of view and truncated, discard and redo the process
        if max(new_fps_2d[:, 0]) >= rgb.shape[1] or max(new_fps_2d[:, 1]) >= rgb.shape[0] or \
                min(new_fps_2d[:, 0]) <= 0 or min(new_fps_2d[:, 1]) <= 0:
            print("Object out of view")
            continue
        # save the data
        cv2.imwrite(output_root + "rgb/" + "%d.jpg" % i, new_rgb)
        cv2.imwrite(output_root + "mask/" + "%d.png" % i, new_mask)
        np.save(output_root + "pose/" + "pose%d.npy" % i, new_pose, allow_pickle=True)
        i += 1
        pbar.update(1)

        # plot and debug
        # debug = 1
        # _, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(mask)
        # axs[0, 1].imshow(rgb)
        # axs[0, 1].plot(fps_2d[:, 0], fps_2d[:, 1], 'go', linewidth=2, markersize=3)
        # axs[0, 1].add_patch(patches.Polygon(xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        # axs[0, 1].add_patch(patches.Polygon(xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        # axs[1, 0].imshow(new_mask)
        # axs[1, 1].imshow(new_rgb)
        # axs[1, 1].plot(new_fps_2d[:, 0], new_fps_2d[:, 1], 'go', linewidth=2, markersize=3)
        # axs[1, 1].add_patch(patches.Polygon(xy=new_corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        # axs[1, 1].add_patch(patches.Polygon(xy=new_corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        # plt.show()