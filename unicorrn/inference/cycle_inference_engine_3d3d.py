import torch

from unicorrn.utils.vision3d.array_ops import denormalize_points_meta


@torch.no_grad()
def cycle_inference(
        sample,
        model,
        src_norm_meta,
        tgt_norm_meta,
        matching_radius=0.02
):
    src_sample = {
        'src_pcd': sample['src_pcd'],
        'tgt_pcd': sample['tgt_pcd'],
        'src_grid_coord': sample['src_grid_coord'],
        'tgt_grid_coord': sample['tgt_grid_coord'],
        'src_length': sample['src_length'],
        'tgt_length': sample['tgt_length']
    }
    tgt_sample = {
        'tgt_pcd': sample['src_pcd'],
        'src_pcd': sample['tgt_pcd'],
        'tgt_grid_coord': sample['src_grid_coord'],
        'src_grid_coord': sample['tgt_grid_coord'],
        'tgt_length': sample['src_length'],
        'src_length': sample['tgt_length']
    }

    pred_forward = model.forward_pcd_to_pcd(sample=src_sample, query_pos=sample['src_pcd'][None])
    output_backward = model.forward_pcd_to_pcd(sample=tgt_sample, query_pos=pred_forward['corr_predictions'])

    src_pcd_raw = denormalize_points_meta(sample['src_pcd'].squeeze(0).cpu(), src_norm_meta)
    output_backward_raw = denormalize_points_meta(output_backward['corr_predictions'].squeeze(0).cpu(), src_norm_meta)
    cycle_matched = torch.norm(output_backward_raw - src_pcd_raw, dim=-1) <= matching_radius
    corr_forward_raw = denormalize_points_meta(pred_forward['corr_predictions'].squeeze(0).cpu(), tgt_norm_meta)

    return corr_forward_raw.detach().cpu().numpy(), cycle_matched.cpu().numpy()
