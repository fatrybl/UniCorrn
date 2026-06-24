import torch


def offset2batch(point, pos3d, offset):
    start = 0
    batch = []
    pos = []
    for stop in offset:
        batch.append(point[start:stop])
        pos.append(pos3d[start:stop])
        start = stop

    return batch, pos


def batch2offset(batch):
    batch = [item.squeeze(0) for item in batch]
    return torch.vstack(batch)


def cuda_attn_bias_padding(attn_bias):
    seq_len_raw = attn_bias.shape[-1]
    if seq_len_raw % 8 != 0:
        seq_len_8 = (seq_len_raw // 8 + 1) * 8
        attn_bias_cuda = torch.zeros((*attn_bias.shape[:-1], seq_len_8), device=attn_bias.device)
        attn_bias_cuda[..., :seq_len_raw] = attn_bias
        attn_bias = attn_bias_cuda[..., :seq_len_raw]

    return attn_bias


def context_attn_bias(context_seq_len, query_seq_len, bidirectional, device='cpu'):
    init_mask = torch.eye(context_seq_len + query_seq_len, dtype=bool, device=device)
    if bidirectional:
        init_mask[:context_seq_len, :] = 1.
    init_mask[:, :context_seq_len] = 1.
    attn_bias = cuda_attn_bias_padding(torch.where(init_mask, 0.0, float('-inf')))

    return attn_bias


def pad_sequences(sequences):
    target_length = max([seq.shape[0] for seq in sequences])
    padded_seq = []
    src_mask = []
    for seq in sequences:
        mask = torch.zeros(target_length, device=seq.device)
        mask[:seq.shape[0]] = 1
        src_mask.append(mask)
        padding = seq.new_zeros(target_length - seq.shape[0], seq.shape[-1])
        padded_seq.append(torch.cat([seq, padding], dim=0))

    return torch.stack(padded_seq), torch.stack(src_mask)


def padding_attn_bias(padded_q_mask, padded_k_mask):
    """
    padded_q_mask - N, Nq
    padded_k_mask - N, Nk
    """
    compound_mask = (padded_q_mask.unsqueeze(-1) @ padded_k_mask.unsqueeze(-2)).to(bool)
    attn_bias = cuda_attn_bias_padding(torch.where(compound_mask, 0.0, float('-inf')))
    return attn_bias


def unpad_sequence(src_sequences, pad_sequences):
    unpad_seq = []
    for original_seq, pad_seq in zip(src_sequences, pad_sequences):
        unpad_seq.append(pad_seq[:original_seq.shape[0]])
    return unpad_seq


def freeze_modules(*modules):
    for m in modules:
        m.eval()
        for params in m.parameters():
            params.requires_grad = False
