'''
Code borrowed from: https://github.com/ubc-vision/COTR/
'''


import random

from collections import namedtuple
from accelerate.state import PartialState
from functools import wraps
from easydict import EasyDict
from networkx.algorithms.flow import maximum_flow_value

from . import debug_utils

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import PIL
import math


'''
ImagePatch: patch: patch content, np array or None
            x: left bound in original resolution
            y: upper bound in original resolution
            w: width of patch
            h: height of patch
            ow: width of original resolution
            oh: height of original resolution
'''
ImagePatch = namedtuple('ImagePatch', ['patch', 'x', 'y', 'w', 'h', 'ow', 'oh'])
Point3D = namedtuple("Point3D", ["id", "arr_idx", "image_ids"])
Point2D = namedtuple("Point2D", ["id_3d", "xy"])


class CropCamConfig():
    def __init__(self, x, y, w, h, out_w, out_h, orig_w, orig_h):
        '''
        xy: left upper corner
        '''
        # assert x > 0 and x < orig_w
        # assert y > 0 and y < orig_h
        # assert w < orig_w and h < orig_h
        # assert x - w / 2 > 0 and x + w / 2 < orig_w
        # assert y - h / 2 > 0 and y + h / 2 < orig_h
        # assert h / w == out_h / out_w
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.out_w = out_w
        self.out_h = out_h
        self.orig_w = orig_w
        self.orig_h = orig_h

    def __str__(self):
        out = f'original image size(h,w): [{self.orig_h}, {self.orig_w}]\n'
        out += f'crop at(x,y):             [{self.x}, {self.y}]\n'
        out += f'crop size(h,w):           [{self.h}, {self.w}]\n'
        out += f'resize crop to(h,w):      [{self.out_h}, {self.out_w}]'
        return out


def fix_randomness(seed=42):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def on_main_process(function):
    """
    Decorator to selectively run the decorated function on the main process only based on the `main_process_only`
    attribute in a class.

    Checks at function execution rather than initialization time, not triggering the initialization of the
    `PartialState`.

    Adapted from https://github.com/huggingface/accelerate/blob/00301b27b75951b6105f2d1a1c4e677a57aba0cd/src/accelerate/tracking.py#L58C1-L75C35
    """

    @wraps(function)
    def execute_on_main_process(self, *args, **kwargs):
        if getattr(self, "main_process_only", False):
            return PartialState().on_main_process(function)(self, *args, **kwargs)
        else:
            return function(self, *args, **kwargs)

    return execute_on_main_process


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def float_image_resize(img, shape, interp=PIL.Image.BILINEAR):
    missing_channel = False
    if len(img.shape) == 2:
        missing_channel = True
        img = img[..., None]
    layers = []
    img = img.transpose(2, 0, 1)
    for l in img:
        l = np.array(PIL.Image.fromarray(l).resize(shape[::-1], resample=interp))
        assert l.shape[:2] == shape
        layers.append(l)
    if missing_channel:
        return np.stack(layers, axis=-1)[..., 0]
    else:
        return np.stack(layers, axis=-1)


def is_nan(x):
    """
    get mask of nan values.
    :param x: torch or numpy var.
    :return: a N-D array of bool. True -> nan, False -> ok.
    """
    return x != x


def has_nan(x) -> bool:
    """
    check whether x contains nan.
    :param x: torch or numpy var.
    :return: single bool, True -> x containing nan, False -> ok.
    """
    if x is None:
        return False
    return is_nan(x).any()


def confirm(question='OK to continue?'):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question + ' [y/n] ').lower()
    return answer == "y"


def print_notification(content_list, notification_type='NOTIFICATION'):
    print('---------------------- {0} ----------------------'.format(notification_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width

    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError('cannot process this image with shape: {0}'.format(np_img.shape))


def safe_load_weights(model, saved_weights):
    try:
        model.load_state_dict(saved_weights)
    except RuntimeError:
        try:
            weights = saved_weights
            weights = {k.replace('module.', ''): v for k, v in weights.items()}
            model.load_state_dict(weights)
        except RuntimeError:
            try:
                weights = saved_weights
                weights = {'module.' + k: v for k, v in weights.items()}
                model.load_state_dict(weights)
            except RuntimeError:
                try:
                    pretrained_dict = saved_weights
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape))}
                    assert len(pretrained_dict) != 0
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    non_match_keys = set(model.state_dict().keys()) - set(pretrained_dict.keys())
                    notification = []
                    notification += ['pretrained weights PARTIALLY loaded, following are missing:']
                    notification += [str(non_match_keys)]
                    print_notification(notification, 'WARNING')
                except Exception as e:
                    print(f'pretrained weights loading failed {e}')
                    exit()
    print('weights safely loaded')


def safe_load_point_transformer_v3_weights(model, saved_weights):
    try:
        weights = {k.replace('module.', ''): v for k, v in saved_weights.items()}
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in weights.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape))}
        assert len(pretrained_dict) != 0
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        non_match_keys = set(model.state_dict().keys()) - set(pretrained_dict.keys())
        notification = []
        notification += ['pretrained weights PARTIALLY loaded, following are missing:']
        notification += [str(non_match_keys)]
        print_notification(notification, 'WARNING')
    except Exception as e:
        print(f'pretrained weights loading failed {e}')
        exit()
    print('weights safely loaded')

def safe_load_pretrained_mapo_wts(model, saved_weights):
    weights = {k.replace('encoder.', 'img_encoder.'): v for k, v in saved_weights.items()}
    safe_load_weights(model, weights)

def visualize_corrs(img1, img2, corrs, mask=None):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)

    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = 400
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = cv2.resize(img1, (0, 0), fx=scale1, fy=scale1)
    img2 = cv2.resize(img2, (0, 0), fx=scale2, fy=scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = .5
    alpha = 1

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[1],
    )
    

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[0],
    )
    plt.scatter(xs, ys)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        #torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(1,2,0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:,:,0] = flow[:,:,0] + X
            map[:,:,1] = flow[:,:,1] + Y
            if output_channel_first:
                map = map.transpose(2,0,1).float()
        return map.astype(np.float32)


def convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            B, C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if map.is_cuda:
                grid = grid.cuda()
            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if map.is_cuda:
                grid = grid.cuda()

            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[3] != 2:
                # size is Bx2xHxW
                map = map.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = map[i, :, :, 0] - X
                flow[i, :, :, 1] = map[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = map.shape[:2]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = map[:,:,0]-X
            flow[:,:,1] = map[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2,0,1).float()
        return flow.astype(np.float32)


def adjust_learning_rate(optimizer, epoch, args):
    """
        Decay the learning rate with half-cycle cosine after warmup
        Borrowed from DUSt3R https://github.com/naver/dust3r
    
    """
    
    if epoch < args.WARMUP_EPOCHS:
        lr = args.LR * epoch / args.WARMUP_EPOCHS 
    else:
        if "DISABLE_LR_DECAY" in args.to_dict() and args.DISABLE_LR_DECAY:
            lr = args.LR
        else:
            lr = args.MIN_LR + (args.LR - args.MIN_LR) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.WARMUP_EPOCHS) / (args.EPOCHS - args.WARMUP_EPOCHS)))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr


def endpointerror(pred, target, dim=-1):
    epe = torch.norm(pred - target, p=2, dim=dim).mean()
    return epe

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):

        self.reset()

    def reset(self):
        """
        Resets the meter
        """

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter

        Parameters
        -----------
        val : float
            Value to update the meter with
        n : int
            Number of samples to update the meter with
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter:
    """
    Computes and stores the average and current value for different keys.
    """

    def __init__(self):
        self.meters = EasyDict()

    def reset(self, key=None):
        """
        Resets the meter for a specific key or all keys if no key is provided.

        Parameters
        -----------
        key : str, optional
            Key to reset. If None, reset all keys.
        """
        if key:
            if key in self.meters:
                self.meters[key] = EasyDict({'val': 0.0, 'avg': 0.0, 'sum': 0.0, 'count': 0})
        else:
            self.meters = EasyDict()

    def update(self, **kwargs):
        """
        Updates the meter for multiple keys using kwargs.

        Parameters
        -----------
        kwargs : dict
            Key-value pairs where the key is a metric name and the value is a tuple (val, n) 
            or just val. If n is not provided, it defaults to 1.
            Example: update(accuracy=(90.0, 2), loss=0.25)
        """
        for key, value in kwargs.items():
            if isinstance(value, tuple):
                val, n = value  # Unpack val and n
            else:
                val, n = value, 1  # If only val is provided, set n to 1

            if key not in self.meters:
                self.meters[key] = EasyDict({'val': 0.0, 'avg': 0.0, 'sum': 0.0, 'count': 0})

            meter = self.meters[key]
            meter['val'] = val
            meter['sum'] += val * n
            meter['count'] += n
            meter['avg'] = meter['sum'] / meter['count']

    def get(self, key):
        """
        Retrieves the current value, sum, count, and average for a specific key.

        Parameters
        -----------
        key : str
            Key to retrieve the values for.
        
        Returns
        -----------
        dict
            Dictionary containing 'val', 'avg', 'sum', and 'count' for the specified key.
        """
        return self.meters.get(key, EasyDict({'val': 0.0, 'avg': 0.0, 'sum': 0.0, 'count': 0}))

    def get_all(self):
        """
        Retrieves the current values for all keys.

        Returns
        -----------
        dict
            Dictionary containing 'val', 'avg', 'sum', and 'count' for all keys.
        """
        return self.meters


def denorm_coords_2d(pred_norm, H, W):
    pred_norm[:, :, 0] = pred_norm[:, :, 0] * W
    pred_norm[:, :, 1] = pred_norm[:, :, 1] * H
    pred_norm = pred_norm.round().int()

    # query coordinates cannot be negative
    mask = pred_norm < 0
    pred_norm[mask] = 0

    return pred_norm

def denorm_coords_3d(pred_norm, pcd, lengths):
    pcd_batch = torch.split(pcd, lengths, dim=0)
    pred_denorm = []
    for pred, pcd in zip(pred_norm, pcd_batch):
        max_val = torch.max(pcd, dim=0).values
        min_val = torch.min(pcd, dim=0).values
        pred_denorm.append(pred * (max_val - min_val) + min_val)

    return torch.stack(pred_denorm)

def denorm_coords_meta_3d(pred_norm, pcd_meta):
    pred_denorm = []
    for pred, meta in zip(pred_norm, pcd_meta):
        pred_center = pred * meta['length']
        pred_denorm.append(pred_center + meta['centroid'])

    return torch.stack(pred_denorm)

def cartesian_img_coord(img_h, img_w, patch_size=None, norm=False):
    if patch_size is None:
        x = torch.arange(img_w, dtype=torch.float32)
        y = torch.arange(img_h, dtype=torch.float32)
        x_norm_factor = img_w - 1
        y_norm_factor = img_h - 1
    else:
        x = torch.arange(img_w, dtype=torch.float32) * patch_size + patch_size / 2
        y = torch.arange(img_h, dtype=torch.float32) * patch_size + patch_size / 2
        x_norm_factor = img_w * patch_size
        y_norm_factor = img_h * patch_size

    if norm:
        x /= x_norm_factor
        y /= y_norm_factor
    cart_prod = torch.cartesian_prod(x, y).reshape(img_w, img_h, 2)
    return cart_prod.permute(1, 0, 2).contiguous()
