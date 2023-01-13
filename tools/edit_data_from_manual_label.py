import json

manual_label_path = "/home/wenbin/clean-pvnet/data/socket_raw/manual_label/"

original_train_path = "/home/wenbin/clean-pvnet/data/custom/train.json"

with open(original_train_path) as f:
    ori_train = json.load(f)

for n in range(10):  # 299
    with open(manual_label_path + str(n) + '.json') as f:
        label = json.load(f)
        # synthesis the new fps_2d as a 7*2 list
        fps_2d = []
        for i in range(7):
            fps_2d.append(label['shapes'][i]['points'][0])
        ori_train['annotations'][n]['fps_2d'] = fps_2d
        debug = 1

with open("/home/wenbin/clean-pvnet/data/custom/train.json", "w") as f:
    json.dump(ori_train, f)