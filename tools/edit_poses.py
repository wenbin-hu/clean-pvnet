import numpy as np
import os

path = "/home/wenbin/clean-pvnet/data/custom/pose/"

for name in os.listdir(path):
    pose = np.load(path + name)
    new_pose = pose[:3, :]
    np.save(path + name, new_pose)