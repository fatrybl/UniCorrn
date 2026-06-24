from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def draw_correspondences(
    img1: np.ndarray,
    img2: np.ndarray,
    queries: np.ndarray,
    targets: np.ndarray,
    output_image: str,
    conf_score: Optional[np.ndarray] = None,
    caption: Optional[str] = None,
):
    """
    img1: H, W, 3
    img2: H, W, 3
    queries: N_seq, 2
    targets: N_seq, 2
    conf_score: N_seq
    """
    assert queries.shape == targets.shape
    if conf_score is not None:
        assert queries.shape[0] == conf_score.shape[0]

    img1_padding = img2.shape[1] // 2
    img2_padding = img1.shape[1] // 2
    img1 = np.pad(img1, ((0, 0), (0, img1_padding), (0, 0)), "constant")
    img2 = np.pad(img2, ((0, 0), (img2_padding, 0), (0, 0)), "constant")

    canvas = np.concatenate([img1, img2], axis=0)
    plt.imshow(canvas)
    target_offset = np.array([img2_padding, img1.shape[0]])
    targets = targets + target_offset

    plt.scatter(queries[:, 0], queries[:, 1], c="red", s=10, marker="x")
    if conf_score is not None:
        cmap = plt.get_cmap("viridis")
        norm = colors.Normalize(vmin=np.min(conf_score), vmax=np.max(conf_score))

        ax = plt.scatter(
            targets[:, 0],
            targets[:, 1],
            c=conf_score,
            cmap=cmap,
            norm=norm,
            s=10,
            marker="x",
        )
        for idx, query in enumerate(queries):
            target = targets[idx]
            plt.plot(
                np.array([query[0], target[0]]),
                np.array([query[1], target[1]]),
                targets[idx],
                color=cmap(norm(conf_score[idx])),
                linewidth=0.5,
            )

        plt.colorbar(ax, label="confidence score")
    else:
        plt.scatter(targets[:, 0], targets[:, 1], c="blue", s=10, marker="x")
        for idx, query in enumerate(queries):
            target = targets[idx]
            plt.plot(
                np.array([query[0], target[0]]),
                np.array([query[1], target[1]]),
                color="green",
                linewidth=0.5,
            )

    if caption is not None:
        plt.title(caption, fontsize=10, color="black")

    plt.savefig(output_image, bbox_inches="tight")
    plt.close()


def plot_correspondences(
    img1,
    img2,
    kpts1,
    kpts2,
    marker_size=5,
    plot_line=True,
    cmap_type="hsv",
    save_path=None,
):
    """
    img1, img2: np.ndarray of shape [H, W, 3]
    kpts1, kpts2: np.ndarray of shape [N, 2] (x, y)
    save_path: optional str, path to save the figure (e.g., 'output.png'). If None, figure is not saved.
    """

    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    # Use the max height so both images fit vertically
    H = max(H1, H2)

    # Create a canvas with both images side by side
    canvas = np.zeros((H, W1 + W2, 3))
    canvas[:H1, :W1, :] = img1
    canvas[:H2, W1:, :] = img2
    canvas = np.clip(canvas / 255, 0, 1)

    # Normalize x-coordinates of kpts1 to [0, 1] for colormap indexing
    x_coords = kpts1[:, 0]
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-8)
    cmap = cm.get_cmap(cmap_type)
    colors = cmap(x_norm)[:, :3]
    plt.figure(figsize=(10, 5))
    plt.imshow(canvas)
    plt.axis("off")

    # Draw correspondences
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(kpts1, kpts2)):
        x2_shifted = x2 + W1  # offset by first image's width, not assumed equal
        plt.scatter([x1, x2_shifted], [y1, y2], color=colors[i], s=marker_size)
        if plot_line:
            plt.plot([x1, x2_shifted], [y1, y2], color=colors[i], linewidth=1)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, pad_inches=0)

    plt.show()
