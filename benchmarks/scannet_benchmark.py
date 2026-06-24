import os.path as osp

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from unicorrn.inference.inference_engine import coarse_only, coarse_to_fine

from .utils import *


class ScanNetBenchmark:
    def __init__(self, data_root="data/scannet", unified_model=False) -> None:
        self.data_root = data_root
        self.unified_model = unified_model

    def benchmark(self, model, query_points, select_layer=-1):
        model.eval()
        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(osp.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            for pairind in tqdm(pair_inds, smoothing=0.9):
                scene = pairs[pairind]
                scene_name = f"scene0{scene[0]}_00"
                im_A_path = osp.join(
                    self.data_root,
                    "scannet_test_1500",
                    scene_name,
                    "color",
                    f"{scene[2]}.jpg",
                )
                im_A = imageio.imread(im_A_path, pilmode="RGB")
                im_B_path = osp.join(
                    self.data_root,
                    "scannet_test_1500",
                    scene_name,
                    "color",
                    f"{scene[3]}.jpg",
                )
                im_B = imageio.imread(im_B_path, pilmode="RGB")
                T_gt = rel_pose[pairind].reshape(3, 4)
                R, t = T_gt[:3, :3], T_gt[:3, 3]
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in open(
                            osp.join(
                                self.data_root,
                                "scannet_test_1500",
                                scene_name,
                                "intrinsic",
                                "intrinsic_color.txt",
                            ),
                            "r",
                        )
                        .read()
                        .split("\n")
                        if r
                    ]
                )
                H1, W1, _ = im_A.shape
                H2, W2, _ = im_B.shape
                K1 = K.copy()
                K2 = K.copy()

                queries = np.array(
                    query_points[f"{pairind}_{scene_name}"]["scaled_kpts1"]
                )

                matched_queries, fine_preds, fine_confidence, _ = coarse_only(
                    im_A,
                    im_B,
                    queries,
                    model,
                    select_layer=select_layer,
                    unified_model=self.unified_model,
                )

                scale1 = 480 / min(W1, H1)
                scale2 = 480 / min(W2, H2)
                w1, h1 = scale1 * W1, scale1 * H1
                w2, h2 = scale2 * W2, scale2 * H2
                K1 = K1 * scale1
                K2 = K2 * scale2

                offset = 0.5

                # Normalize keypoints between 0 and 1 and then scale them.
                kpts1 = matched_queries
                kpts1 = np.stack(
                    (
                        w1 * (kpts1[:, 0] / W1) - offset,
                        h1 * (kpts1[:, 1] / H1) - offset,
                    ),
                    axis=-1,
                )

                kpts2 = fine_preds
                kpts2 = np.stack(
                    (
                        w2 * (kpts2[:, 0] / W2) - offset,
                        h2 * (kpts2[:, 1] / H2) - offset,
                    ),
                    axis=-1,
                )

                for _ in range(5):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]
                    try:
                        norm_threshold = 0.5 / (
                            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                        )
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        )
                        T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as e:
                        print(repr(e))
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
