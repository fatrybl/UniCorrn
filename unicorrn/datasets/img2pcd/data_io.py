import os
import os.path as osp
import pickle
from typing import List, Optional

import torch
import torchvision.transforms.v2 as transforms
import numpy as np
from numpy import ndarray
from PIL import Image
from skimage import io
from tqdm import tqdm


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# reading / writing


def readlines(filename: str) -> List[str]:
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def writelines(lines: List[str], filename: str, mode: str = "w"):
    lines = [line + "\n" if not line.endswith("\n") else line for line in lines]
    with open(filename, mode) as f:
        f.writelines(lines)


def read_image(filename: str, as_gray: bool = False) -> ndarray:
    image = io.imread(filename, as_gray=as_gray)
    # if not as_gray:
    #     image = image.astype(np.float32) / 255.0
    return image


def read_depth_image(filename: str, scaling_factor: Optional[float] = None) -> ndarray:
    image = io.imread(filename).astype(np.float32)
    if scaling_factor is not None:
        image = image / scaling_factor
    return image


# log utilities


def read_log(line):
    split_line = line.split(", ")
    data_dict = {}
    metadata = []
    for item in split_line:
        if ": " not in item:
            metadata.append(item)
        else:
            key, value = item.split(": ")
            data_dict[key] = value
    data_dict["metadata"] = metadata
    return data_dict


def read_logs(log_file):
    lines = readlines(log_file)
    log_dicts = [read_log(line) for line in lines]
    return log_dicts


def write_correspondences(file_name: str, src_corr_points: ndarray, tgt_corr_points: ndarray):
    if not file_name.endswith(".obj"):
        file_name += ".obj"

    v_lines = []
    l_lines = []

    num_corr = src_corr_points.shape[0]
    for i in tqdm(range(num_corr)):
        n = i * 2

        src_point = src_corr_points[i]
        tgt_point = tgt_corr_points[i]

        line = "v {:.6f} {:.6f} {:.6f}\n".format(src_point[0], src_point[1], src_point[2])
        v_lines.append(line)

        line = "v {:.6f} {:.6f} {:.6f}\n".format(tgt_point[0], tgt_point[1], tgt_point[2])
        v_lines.append(line)

        line = "l {} {}\n".format(n + 1, n + 2)
        l_lines.append(line)

    with open(file_name, "w") as f:
        f.writelines(v_lines)
        f.writelines(l_lines)


def get_transform(color_jitter):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_list = []
    if color_jitter:
        transform_list.append(transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ))
    transform_list.append(transforms.ToDtype(torch.float32, scale=True))
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)
