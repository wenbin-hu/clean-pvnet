import json

fps_label_path = "/home/wenbin/clean-pvnet/data/custom/fps_2d.json"

original_train_path = "/home/wenbin/clean-pvnet/data/custom/train.json"

with open(original_train_path) as f:
    ori_train = json.load(f)

with open(fps_label_path) as f:
    fps_2d = json.load(f)

for n in range(len(fps_2d)):
    ori_train['annotations'][n]['fps_2d'] = fps_2d[n]
    debug = 1

with open("/home/wenbin/clean-pvnet/data/custom/train.json", "w") as f:
    json.dump(ori_train, f)