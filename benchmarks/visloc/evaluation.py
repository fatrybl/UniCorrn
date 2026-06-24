# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# evaluation utilities
# --------------------------------------------------------
import collections
import os

import numpy as np
import quaternion
import torch


def aggregate_stats(info_str, pose_errors, angular_errors):
    stats = collections.Counter()
    median_pos_error = np.median(pose_errors)
    median_angular_error = np.median(angular_errors)
    out_str = f"{info_str}: {len(pose_errors)} images - {median_pos_error=}, {median_angular_error=}"

    for trl_thr, ang_thr in [(0.1, 1), (0.25, 2), (0.5, 5), (5, 10)]:
        for pose_error, angular_error in zip(pose_errors, angular_errors):
            correct_for_this_threshold = (pose_error < trl_thr) and (
                angular_error < ang_thr
            )
            stats[trl_thr, ang_thr] += correct_for_this_threshold
    stats = {
        f"acc@{key[0]:g}m,{key[1]}deg": 100 * val / len(pose_errors)
        for key, val in stats.items()
    }
    for metric, perf in stats.items():
        out_str += f"  - {metric:12s}={float(perf):.3f}"
    return out_str


_ONE_OVER_2SQRT2 = 1.0 / (2 * np.sqrt(2))


def rotmat_geodesic_distance(R1, R2, clamping=1.0):
    """

    #RoMa
    # Copyright (c) 2020 NAVER Corp.
    # 3-Clause BSD License.
    Returns the angular distance alpha between a pair of rotation matrices.
    Based on the equality :math:`|R_2 - R_1|_F = 2 \sqrt{2} sin(alpha/2)`.

    Args:
        R1, R2 (...x3x3 tensor): batch of 3x3 rotation matrices.
        clamping: clamping value applied to the input of :func:`torch.asin()`.
                Use 1.0 to ensure valid angular distances.
                Use a value strictly smaller than 1.0 to ensure finite gradients.
    Returns:
        batch of angles in radians (... tensor).
    """
    return 2.0 * torch.asin(
        torch.clamp_max(torch.norm(R2 - R1, dim=[-1, -2]) * _ONE_OVER_2SQRT2, clamping)
    )


def get_pose_error(pr_camtoworld, gt_cam_to_world):
    abs_transl_error = torch.linalg.norm(
        torch.tensor(pr_camtoworld[:3, 3]) - torch.tensor(gt_cam_to_world[:3, 3])
    )
    abs_angular_error = (
        rotmat_geodesic_distance(
            torch.tensor(pr_camtoworld[:3, :3]), torch.tensor(gt_cam_to_world[:3, :3])
        )
        * 180
        / np.pi
    )
    return abs_transl_error, abs_angular_error


def export_results(output_dir, xp_label, query_names, poses_pred):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        lines = ""
        lines_ltvl = ""
        for query_name, pr_querycam_to_world in zip(query_names, poses_pred):
            if pr_querycam_to_world is None:
                pr_world_to_querycam = np.eye(4)
            else:
                pr_world_to_querycam = np.linalg.inv(pr_querycam_to_world)
            query_shortname = os.path.basename(query_name)
            pr_world_to_querycam_q = quaternion.from_rotation_matrix(
                pr_world_to_querycam[:3, :3]
            )
            pr_world_to_querycam_t = pr_world_to_querycam[:3, 3]

            line_pose = (
                quaternion.as_float_array(pr_world_to_querycam_q).tolist()
                + pr_world_to_querycam_t.flatten().tolist()
            )

            line_content = [query_name] + line_pose
            lines += " ".join(str(v) for v in line_content) + "\n"

            line_content_ltvl = [query_shortname] + line_pose
            lines_ltvl += " ".join(str(v) for v in line_content_ltvl) + "\n"

        with open(os.path.join(output_dir, xp_label + "_results.txt"), "wt") as f:
            f.write(lines)
        with open(os.path.join(output_dir, xp_label + "_ltvl.txt"), "wt") as f:
            f.write(lines_ltvl)
