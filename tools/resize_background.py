import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

ori_path = "/home/wenbin/clean-pvnet/data/background"
new_size = (640, 480)

num = len([name for name in os.listdir(ori_path) if os.path.isfile(os.path.join(ori_path, name))])
for name in os.listdir(ori_path):
    img = cv2.imread(ori_path + '/' + name)
    new_img = cv2.resize(img, new_size)
    cv2.imwrite(ori_path + '/' + name, new_img)

debug = 1