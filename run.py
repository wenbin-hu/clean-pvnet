from lib.config import cfg, args
import numpy as np
import os
import matplotlib.pyplot as plt
from lib.datasets.dataset_catalog import DatasetCatalog
import pycocotools.coco as coco
from lib.utils.pvnet import pvnet_config
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils


def run_rgb():
    import glob
    from scipy.misc import imread
    import matplotlib.pyplot as plt

    syn_ids = sorted(os.listdir('data/ShapeNet/renders/02958343/'))[-10:]
    for syn_id in syn_ids:
        pkl_paths = glob.glob('data/ShapeNet/renders/02958343/{}/*.pkl'.format(syn_id))
        np.random.shuffle(pkl_paths)
        for pkl_path in pkl_paths:
            img_path = pkl_path.replace('_RT.pkl', '.png')
            img = imread(img_path)
            plt.imshow(img)
            plt.show()


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'], batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    torch.manual_seed(0)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

# python run.py --type visualize --cfg_file configs/custom.yaml
def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    cnt = 0
    for batch in data_loader:
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        # visualizer.visualize(output, batch)
        visualizer.visualize_gt(output, batch, cnt)
        cnt += 1


def run_visualize_train():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=True)
    visualizer = make_visualizer(cfg, 'train')
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize_train(output, batch)


def run_analyze():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.analyzers import make_analyzer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    analyzer = make_analyzer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        analyzer.analyze(output, batch)


def run_net_utils():
    from lib.utils import net_utils
    import torch
    import os

    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    pretrained_model = torch.load(model_path)
    net = pretrained_model['net']
    net = net_utils.remove_net_prefix(net, 'dla.')
    net = net_utils.remove_net_prefix(net, 'cp.')
    pretrained_model['net'] = net
    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    os.system('mkdir -p {}'.format(os.path.dirname(model_path)))
    torch.save(pretrained_model, model_path)


def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


def run_tless():
    from lib.datasets.tless import handle_rendering_data, fuse, handle_test_data, handle_ag_data, tless_to_coco
    # handle_rendering_data.render()
    # handle_rendering_data.render_to_coco()
    # handle_rendering_data.prepare_asset()

    # fuse.fuse()
    # handle_test_data.get_mask()
    # handle_test_data.test_to_coco()
    handle_test_data.test_pose_to_coco()

    # handle_ag_data.ag_to_coco()
    # handle_ag_data.get_ag_mask()
    # handle_ag_data.prepare_asset()

    # tless_to_coco.handle_train_symmetry_pose()
    # tless_to_coco.tless_train_to_coco()


def run_ycb():
    from lib.datasets.ycb import handle_ycb
    handle_ycb.collect_ycb()


def run_render():
    from lib.utils.renderer import opengl_utils
    from lib.utils.vsd import inout
    from lib.utils.linemod import linemod_config
    import matplotlib.pyplot as plt

    obj_path = 'data/linemod/cat/cat.ply'
    model = inout.load_ply(obj_path)
    model['pts'] = model['pts'] * 1000.
    im_size = (640, 300)
    opengl = opengl_utils.NormalRender(model, im_size)

    K = linemod_config.linemod_K
    pose = np.load('data/linemod/cat/pose/pose0.npy')
    depth = opengl.render(im_size, 100, 10000, K, pose[:, :3], pose[:, 3:] * 1000)

    plt.imshow(depth)
    plt.show()


# pre-process the custom data-set
def run_custom():
    from tools import handle_custom_dataset
    data_root = 'data/custom'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)


def run_detector_pvnet():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)

# python run.py --type demo --cfg_file configs/linemod.yaml demo_path demo_images/cat
def run_demo():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    from tqdm import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import cv2

    torch.manual_seed(0)
    # meta includes: 'kpt_3d', 'corner_3d', 'K' keypoints and corners in the 3D model
    # K is the camera intrinsic matrix 3x3
    meta = np.load(os.path.join(cfg.demo_path, '../meta.npy'), allow_pickle=True).item()
    demo_images = glob.glob(cfg.demo_path + '/*jpg')

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean = pvnet_config.mean
    std = pvnet_config.std
    # plt.ion()
    # figure, ax = plt.subplots()
    cnt = 0
    for demo_image in tqdm(demo_images):
        demo_image = np.array(Image.open(demo_image)).astype(np.float32)
        demo_image = cv2.resize(demo_image, (640, 480))  # resize to [640, 480]
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            # output includes: 'seg', 'vertex', 'mask', 'kpt_2d'
            output = network(inp)
        # visualizer.visualize_demo_auto(output, inp, meta, figure, ax)
        visualizer.visualize_demo_single(output, inp, meta, cnt)
        cnt += 1

# for analysing the process, testing the computation time
# python run.py --type test --cfg_file configs/custom.yaml demo_path demo_images/cat
def run_test():
    from tqdm import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import time
    import cv2

    torch.manual_seed(0)
    # meta includes: 'kpt_3d', 'corner_3d', 'K' keypoints and corners in the 3D model
    # K is the camera intrinsic matrix 3x3
    meta = np.load(os.path.join(cfg.demo_path, '../meta.npy'), allow_pickle=True).item()
    demo_images = glob.glob(cfg.demo_path + '/*jpg')

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    mean = pvnet_config.mean
    std = pvnet_config.std
    plt.ion()
    figure, ax = plt.subplots()
    time_list = []
    for demo_image in tqdm(demo_images):
        demo_image = np.array(Image.open(demo_image)).astype(np.float32)
        st = time.time()
        demo_image = cv2.resize(demo_image, (640, 480))  # resize to [640, 480]
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            # output includes: 'seg', 'vertex', 'mask', 'kpt_2d'
            output = network(inp)
        inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        kpt_3d = np.array(meta['kpt_3d'])
        K = np.array(meta['K'])
        # PnP
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        time_list.append(time.time() - st)
        corner_3d = np.array(meta['corner_3d'])
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        # plot
        ax.imshow(inp)
        patch0 = ax.add_patch(
            patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        patch1 = ax.add_patch(
            patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        figure.canvas.draw()
        figure.canvas.flush_events()
        patch0.remove()
        patch1.remove()
    print(time_list)
    print("average time: ", np.mean(time_list))

if __name__ == '__main__':
    globals()['run_'+args.type]()

