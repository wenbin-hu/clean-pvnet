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
    i = 888
    rgb = cv2.imread(rgb_dir + '/%d.jpg' % i)
    mask = cv2.imread(mask_dir + '/%d.png' % i)
    pose = np.load(pose_dir + '/pose%d.npy' % i, allow_pickle=True)
    corner_2d = pvnet_pose_utils.project(corner_3d, K, pose)
    fps_2d = pvnet_pose_utils.project(fps_3d, K, pose)
    
    # if the object is out of view and truncated, discard and redo the process
    if max(fps_2d[:, 0]) >= rgb.shape[1] or max(fps_2d[:, 1]) >= rgb.shape[0] or \
            min(fps_2d[:, 0]) < 1 or min(fps_2d[:, 1]) < 1:
        print("Object out of view")
    
    # plot
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(mask)
    axs[1].imshow(rgb)
    axs[1].plot(fps_2d[:, 0], fps_2d[:, 1], 'go', linewidth=2, markersize=3)
    axs[1].add_patch(patches.Polygon(xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
    axs[1].add_patch(patches.Polygon(xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
    plt.show()