import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
from handle_custom_dataset import get_model_corners

if __name__ == '__main__':
    # path
    data_root = "/home/wenbin/clean-pvnet/data/custom/"
    pose_dir = os.path.join(data_root, 'pose')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')
    model_dir = os.path.join(data_root, 'model.ply')

    # load the original data
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))
    renderer = OpenGLRenderer(model_dir)
    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)

    # load the data
    for i in range(1200):
        bgr = cv2.imread(rgb_dir + '/%d.jpg' % i)
        mask = cv2.imread(mask_dir + '/%d.png' % i)
        _, mask = cv2.threshold(mask, 5, 1, cv2.THRESH_BINARY)  # binarize the mask
        mask = mask[:, :, 0]
        pose = np.load(pose_dir + '/pose%d.npy' % i, allow_pickle=True)
        corner_2d = pvnet_pose_utils.project(corner_3d, K, pose).astype(int)
        fps_2d = pvnet_pose_utils.project(fps_3d, K, pose).astype(int)
        # plot with opencv
        line_order = [0, 2, 6, 4, 0, 1, 5, 7, 3, 1, 3, 2, 6, 7, 5, 4]
        for point in fps_2d:
            cv2.circle(bgr, tuple(point), 2, (0, 255, 0), -1)
        for i in range(len(line_order) - 1):
            cv2.line(bgr, tuple(corner_2d[line_order[i], :]),
                            tuple(corner_2d[line_order[i + 1], :]), (0, 0, 0), 2)
        final_frame = cv2.hconcat((bgr, cv2.bitwise_and(bgr, bgr, mask=mask)))
        cv2.imshow('training data', final_frame)
        cv2.waitKey(0)
        # plot with matplotlib
        # _, axs = plt.subplots(1, 2)
        # axs[0].imshow(mask)
        # axs[1].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # axs[1].plot(fps_2d[:, 0], fps_2d[:, 1], 'go', linewidth=2, markersize=3)
        # axs[1].add_patch(patches.Polygon(xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        # axs[1].add_patch(patches.Polygon(xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        # plt.show()