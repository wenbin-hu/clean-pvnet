import cv2
import numpy as np
import random
import os
from tqdm import tqdm

# resize and load the background
bgs = []
bg_path = "/home/wenbin/clean-pvnet/data/background/"
tmp_num = len([name for name in os.listdir(bg_path) if os.path.isfile(os.path.join(bg_path, name))])
for name in os.listdir(bg_path):
    img = cv2.imread(bg_path + name)
    new_img = cv2.resize(img, (640, 480))
    bgs.append(new_img)

data_path = "/home/wenbin/clean-pvnet/data/custom/"
# load the original mug pictures
ori_mugs = []
rgb_path = data_path + "rgb/"
for i in range(len(os.listdir(rgb_path))):
    ori_mugs.append(cv2.imread(rgb_path + str(i) + '.jpg'))

# load the masks
ori_masks = []
mask_path = data_path + "mask/"
for i in range(len(os.listdir(mask_path))):
    ori_masks.append(cv2.imread(mask_path + str(i) + '.png'))

# load the poses
ori_poses = []
pose_path = data_path + "pose/"
for i in range(len(os.listdir(pose_path))):
    ori_poses.append(np.load(pose_path + 'pose%d.npy' % i))

cnt = 0
for i in tqdm(range(len(ori_mugs))):
    mug = ori_mugs[i]
    mask = ori_masks[i]
    pose = ori_poses[i]
    # save the original image 25%
    if np.random.rand() > 0.75:
        cv2.imwrite(rgb_path + "%d.jpg" % cnt, mug)
        cv2.imwrite(mask_path + "%d.png" % cnt, mask)
        np.save(pose_path + "pose%d.npy" % cnt, pose)
        cnt += 1
    else:
        # save the image with new background 75%
        for bg in random.sample(bgs, k=1):
            mug_idxs = np.where(mask[:, :, 0] == 255)
            new_bg = bg.copy()
            new_bg[mug_idxs[0], mug_idxs[1], :] = mug[mug_idxs[0], mug_idxs[1], :]
            cv2.imwrite(rgb_path + "%d.jpg"%cnt, new_bg)
            cv2.imwrite(mask_path + "%d.png"%cnt, mask)
            np.save(pose_path + "pose%d.npy"%cnt, pose)
            cnt += 1
            # showing for debug
            # cv2.imshow('Image with new background', new_bg)
            # cv2.waitKey(0)