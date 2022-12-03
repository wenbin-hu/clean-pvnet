import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# load the background
bgs = []
bg_path = "/home/wenbin/clean-pvnet/data/background/"
tmp_num = len([name for name in os.listdir(bg_path) if os.path.isfile(os.path.join(bg_path, name))])
for name in os.listdir(bg_path):
    bgs.append(cv2.imread(bg_path + name))

data_path = "/home/wenbin/clean-pvnet/data/custom/"
# load the original mug pictures
ori_mugs = []
mug_path = data_path + "rgb/"
for i in range(600):
    ori_mugs.append(cv2.imread(mug_path + str(i) + ".jpg"))

# load the masks
ori_masks = []
mask_path = data_path + "mask/"
for i in range(600):
    ori_masks.append(cv2.imread(mask_path + str(i) + ".png"))

# load the poses
ori_poses = []
pose_path = data_path + "pose/"
for i in range(600):
    ori_poses.append(np.load(pose_path + str(i) + ".npy"))

cnt = 0
for i in range(len(ori_mugs)):
    mug = ori_mugs[i]
    mask = ori_masks[i]
    pose = ori_poses[i]
    # save the original images
    cv2.imwrite(data_path + "rgb/%d.jpg" % cnt, mug)
    cv2.imwrite(mask_path + "%d.png" % cnt, mask)
    np.save(pose_path + "pose%d.npy" % cnt, pose)
    cnt += 1
    for bg in random.sample(bgs, k=5):
        mug_idxs = np.where(mask[:, :, 0] == 255)
        new_bg = bg.copy()
        new_bg[mug_idxs[0], mug_idxs[1], :] = mug[mug_idxs[0], mug_idxs[1], :]
        cv2.imwrite(data_path + "rgb/%d.jpg"%cnt, new_bg)
        cv2.imwrite(mask_path + "%d.png"%cnt, mask)
        np.save(pose_path + "pose%d.npy"%cnt, pose)
        cnt += 1
        plt.imshow(new_bg)
        plt.show()
        # debug = 1