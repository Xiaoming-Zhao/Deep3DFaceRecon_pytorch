"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
import tqdm
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat


def get_data_path(img_root, detect_root):

    if os.path.exists(os.path.join(detect_root, "fail_list.txt")):
        with open(os.path.join(detect_root, "fail_list.txt"), "r") as f:
            fail_list = [_.strip() for _ in f.readlines()]
    else:
        fail_list = []

    all_im_path = [
        os.path.join(img_root, i)
        for i in sorted(os.listdir(img_root))
        if i.endswith("png") or i.endswith("jpg")
    ]
    # filter out failing cases
    im_path = []
    for elem in all_im_path:
        if os.path.basename(elem) not in fail_list:
            im_path.append(elem)
    print(f"\nFind {len(im_path)} valid images from {len(all_im_path)} images.\n")

    lm_path = [i.replace("png", "txt").replace("jpg", "txt") for i in im_path]
    lm_path = [
        os.path.join(detect_root, "results", os.path.basename(i)) for i in lm_path
    ]

    return im_path, lm_path


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert("RGB")
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = (
            torch.tensor(np.array(im) / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def main(rank, opt, img_root, detect_root):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(img_root, detect_root)
    lm3d_std = load_lm3d(opt.bfm_folder)

    save_dir = os.path.join(
        visualizer.img_dir, os.path.basename(img_root), "epoch_%s_%06d" % (opt.epoch, 0)
    )
    os.makedirs(save_dir, exist_ok=True)

    BATCH_N = 20

    for i in tqdm.tqdm(range(len(im_path))):
        # print(i, im_path[i])
        img_name = (
            im_path[i].split(os.path.sep)[-1].replace(".png", "").replace(".jpg", "")
        )
        # if not os.path.isfile(lm_path[i]):
        #     continue
        assert os.path.isfile(lm_path[i]), lm_path[i]
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {"imgs": im_tensor, "lms": lm_tensor}
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        model.save_coeff(
            os.path.join(save_dir, img_name + ".mat")
        )  # save predicted coefficients


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options

    img_root = os.path.join(opt.gmpi_root, "runtime_dataset/metfaces1024x1024_xflip")
    detect_root = os.path.join(opt.gmpi_root, "runtime_dataset/metfaces_detect/detections")

    main(0, opt, img_root, detect_root)
