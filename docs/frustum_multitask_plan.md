# UniCorrn Multi-Task Extension — Matching + Confidence + Geometric Frustum Classification

Status: **implemented** on branch `frustum-multitask` (fork `UniCorrn-Frustum`).
Sections 1–2 give the theory; **Section 12 is the authoritative as-built description** and
supersedes the earlier exploratory proposal (notably: the model takes **no camera parameters as
input** — see updated §1.4 — and frustum queries are a **separate** query set, so no masking of
the matching losses is required).
Author intent: extend the existing 2D↔3D matcher so that, in the **point-cloud → image**
direction, every 3D point query produces three outputs:

1. **Matching** — predicted image coordinate of the point (existing `corr`).
2. **Confidence** — DUSt3R-style reliability of that match (existing `info`).
3. **Frustum membership** — binary "is this point inside the camera viewing cone" (new).

Definition used throughout: **(a) geometric frustum membership** — the point lies in the
camera viewing cone given intrinsics + the (unknown-at-inference) extrinsic, *regardless of
occlusion or texture*. This is **not** co-visibility.

---

## 1. Why this task, and why it must be extrinsic-free

### 1.1 Motivation: the 1/Z² depth-observability problem

Calibration / registration that minimises a **reprojection** cost is weakly observable in the
forward (Z) direction. For a point `(X, Y, Z)` in camera frame with focal `f`, principal point
`c_x`:

```
u = f * X / Z + c_x
```

Under a forward/backward translation `t_z` (`Z → Z + t_z`):

```
∂u/∂t_z = − f * X / (Z + t_z)²  =  − (u − c_x) / (Z + t_z)
```

So the pixel position's sensitivity to depth translation **decays as 1/Z²** and additionally
**vanishes for on-axis points** (`u ≈ c_x`). Far points contribute almost no Z-gradient — they
project to nearly the same pixel whether moved several centimetres forward or backward. This is
the well-known weak-Z-observability of projection-based alignment.

### 1.2 Why frustum membership restores Z observability

Frustum membership has a **fundamentally different functional dependence on `t_z`**.

Horizontal FOV half-angle `θ_x`. A point is inside the horizontal cone iff
`|X| ≤ Z · tan(θ_x/2)`. Define the signed metric margin:

```
g(t_z) = (Z + t_z) · tan(θ_x/2) − |X|        ⇒   ∂g/∂t_z = tan(θ_x/2)   (constant in Z)
```

Near-plane margin (point in front of camera):

```
h(t_z) = (Z + t_z) − z_near                  ⇒   ∂h/∂t_z = 1            (constant in Z)
```

The frustum **decision boundary moves with O(1) sensitivity to `t_z`, independent of depth.**
A soft membership `m(T) = σ(g/τ) · σ(h/τ) · (vertical term)` therefore has a **non-vanishing
Z-gradient for any point near the boundary, at any range** — exactly the regime where
reprojection is blind.

Trade-off (stated honestly): this O(1) signal is contributed only by points **near** the cone
boundary (where `σ'` is appreciable); deep-interior and far-exterior points saturate and
contribute nothing. So frustum membership is a **sparse but depth-unbiased** Z signal, meant to
**complement** (not replace) reprojection, which uses all points but is depth-biased.

### 1.3 Downstream use (why we build it)

At calibration time the extrinsic `T` is unknown — it is what we optimise. The network supplies
a **T-independent target** `p_i = P(point i ∈ frustum | cloud, image, intrinsics)`. We then
minimise a consistency term over `T`:

```
L_z(T) = Σ_i  BCE( m_i(T), p_i )
```

where `m_i(T)` is the *differentiable geometric* membership under the current `T` (Section 1.2)
and `p_i` is the *fixed, learned* membership. `∂L_z/∂t_z` aggregates O(1) contributions from
boundary points → adds the Z observability reprojection lacks.

### 1.4 Design decision — NO camera parameters as input (T- and K-independent)

This is the load-bearing decision, and it was strengthened during implementation:

- **Do NOT feed the extrinsic `T`.** If the classifier sees `T`, geometric membership collapses
  to a deterministic reprojection (`transform → project → bounds check`); `p_i` would depend on
  the very `T` we optimise downstream, and `L_z` would be circular with **zero** gradient. ❌
- **Do NOT feed the intrinsics `K` either.** The decoder already regresses the query's projection
  into the *actual image*, whose extent and content embody `K`. Membership ≈ "predicted
  projection lands inside the real image and in front" — so `K` is available *implicitly* through
  the image. Omitting it keeps the model maximally robust and generalisable, and consistent with
  the rest of UniCorrn (which takes no camera params). ✅

`T` and `K` are used **only** to generate labels offline (Section 12 / `frustum_labels.py`). The
network's inputs are exactly those of matching: point cloud + image. This is what makes the
learned membership `p_i` an *independent* target usable to drive `T` (and especially `t_z`).

---

## 2. UniCorrn today (recap of the relevant path)

`pcd → img` flow (`unicorrn/model/modules/unified_query_decoder.py::forward_pcd_to_img`):

- Sample per-query 3D descriptors → `q` (appearance stream init).
- Iterate `DualStreamQueryDecoderBlock` × `dec_depth`:
  - appearance stream `q`, positional stream `hidden_state`, shared Gaussian attention `A`.
- Outputs:
  - `corr = self.corr_embed_2d(hidden_state)` — predicted **image coordinate** (pseudo-inverse
    of the invertible positional encoding).
  - `info = self.info_embed(q)` — confidence logit.
  - `gm_out` — per-layer raw-coordinate soft-argmax (deep supervision).

Key facts that shape this plan:

- The confidence head is a **reliability regressor** (`loss * conf − α·log conf`), **not** a
  matchability classifier (`unicorrn/trainer/functions.py::ConfidenceMatchingLoss`).
- **All queries are assumed valid** — they are sampled from GT correspondences, and
  `UnifiedInfoNCELoss` hard-codes `valid = ones`. There are **no negatives** anywhere today.
  Introducing out-of-frustum negatives is the main pipeline change.

---

## 3. Architecture modifications

### 3.1 Intrinsics conditioning

Encode normalised intrinsics into a conditioning vector added to the query initialisation.

`unicorrn/model/embedder/` (new small module, or fold into the decoder):

```python
class IntrinsicsEncoder(nn.Module):
    """Encode normalised camera intrinsics into a conditioning embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = Mlp(6, hidden_features=dim, out_features=dim)

    def forward(self, fx, fy, cx, cy, width, height) -> Tensor:
        """Map FOV-normalised intrinsics to a per-sample conditioning vector."""
        feats = torch.stack(
            [fx / width, fy / height, cx / width, cy / height,
             2 * torch.atan(0.5 * width / fx), 2 * torch.atan(0.5 * height / fy)],
            dim=-1,
        )
        return self.mlp(feats)
```

Inject at query init in `forward_pcd_to_img` / `forward_img_to_pcd`:

```python
q = src_desc.clone() + self.intrinsics_encoder(*cam_params)[:, None, :]
```

Alternative (stronger): FiLM modulation of each decoder block from this embedding. Start with
the additive form; upgrade only if frustum accuracy near the FOV boundary stalls.

### 3.2 Frustum classification head

Tap **all three signals** — appearance `q`, geometric positional state `hidden_state`, and the
predicted projection `corr`. For definition (a), the geometric streams matter most: a textureless
in-cone point has no appearance match but still has a valid predicted projection.

`unicorrn/model/modules/unified_query_decoder.py`:

```python
class FrustumHead(nn.Module):
    """Binary geometric frustum-membership head for point queries."""

    def __init__(self, feat_dim: int, pos_dim: int, coord_dim: int) -> None:
        super().__init__()
        self.mlp = Mlp(feat_dim + pos_dim + coord_dim, hidden_features=feat_dim,
                       out_features=1)

    def forward(self, q: Tensor, hidden_state: Tensor, corr: Tensor) -> Tensor:
        """Predict in/out-frustum logits from appearance, geometry and projection."""
        return self.mlp(torch.cat([q, hidden_state, corr], dim=-1))
```

Wire it into `QueryMatchingDecoder.__init__`:

```python
self.frustum_head = FrustumHead(project_dim, pos_embed_dim, coord_dim=2)
```

and return its output at the end of `forward_pcd_to_img` (after `corr` / `info` are computed):

```python
frustum = self.frustum_head(q, hidden_state, corr)
return corr, info, frustum, q, src_desc, tgt_desc, gm_out
```

(For `forward_img_to_pcd`, `coord_dim=3`; keep a 3D `frustum_head` or pad as the decoder already
pads `gm_res`.)

### 3.3 Model-level changes

`unicorrn/model/unicorrn.py`:

- `forward_pcd_to_img` accepts `cam_params` and threads them to the decoder.
- Surface `frustum_predictions` in the output dict.
- Allow **dense queries**: today queries are a sampled subset. For classification, query all
  points (or a large balanced subset). Decoder cost is `O(#pts × #img_tokens × depth)`; subsample
  if memory-bound, or run the light variant (Section 7).

```python
return {
    "corr_predictions": out,
    "info_predictions": info,
    "frustum_predictions": frustum,
    ...
}
```

---

## 4. Data pipeline modifications

`unicorrn/datasets/img2pcd/`.

### 4.1 Negative (out-of-frustum) sampling

Currently only in-frustum, GT-corresponded points are sampled. Add a configurable fraction of
**out-of-frustum negatives** per sample (e.g. 50/50, then tune for imbalance). Negatives carry a
frustum label `0` and **no** matching/confidence target.

### 4.2 Label generation (offline, from GT `K, T`)

For each sampled point `P` (LiDAR/world frame):

```python
P_cam = R @ P + t                      # GT extrinsic
in_front = P_cam[2] > z_near
u = fx * P_cam[0] / P_cam[2] + cx
v = fy * P_cam[1] / P_cam[2] + cy
in_bounds = (0 <= u < W) and (0 <= v < H)
frustum_label = in_front and in_bounds     # definition (a), no depth/occlusion test
```

Optional far plane: `P_cam[2] < z_far`. No z-buffer test — (a) ignores occlusion by design.

### 4.3 Batch fields

Add per query:

- `frustum_label: (B, Nq)` — float {0,1}.
- `valid_match: (B, Nq)` — bool; `True` only for in-frustum points that also have a GT
  correspondence (drives masking of matching/confidence/InfoNCE).
- `cam_params` — `fx, fy, cx, cy, W, H` per sample.

---

## 5. Loss design

### 5.1 Masking principle (DETR-style)

Treat it exactly like DETR class-vs-box: the **classification** loss is applied to **all** queries;
the **regression / confidence / contrastive** losses are applied **only to positives**
(`valid_match`). Out-of-frustum points have no target and must be masked out of `corr`/`info`/
InfoNCE — otherwise they inject meaningless regression targets.

Concretely:

- `AuxiliaryGlobalMatchingLoss` (`unified_functions.py`): multiply per-query `loss` and `conf_loss`
  by `valid_match` before reduction (currently unmasked — must change).
- `UnifiedInfoNCELoss`: replace `valid = ones` with the real `valid_match` mask (the
  `valid_matches` argument already exists in `InfoNCE.__call__`).

### 5.2 Frustum loss (new)

Heavy class imbalance is expected (LiDAR FOV ≫ camera FOV). Use focal BCE.

```python
class FrustumClassificationLoss:
    """Focal BCE for geometric frustum membership over all point queries."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        self.gamma = gamma
        self.pos_weight = pos_weight

    def __call__(self, logits, labels):
        """Focal binary cross-entropy between frustum logits and {0,1} labels."""
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=logits.device))
        focal = (1 - torch.where(labels > 0.5, p, 1 - p)) ** self.gamma
        return (focal * ce).mean()
```

Optional: per-decoder-layer deep supervision (mirror the `gamma` decay used for `gm_intermediates`)
by adding a frustum head readout per layer; defer unless single-readout underfits boundaries.

### 5.3 Total objective

```
L = w_match · L_match(masked to valid_match)
  + L_conf (masked to valid_match)            # confidence already couples to L_match
  + w_feat  · L_infonce(masked to valid_match)
  + w_cls   · L_frustum(all queries)
```

Suggested start: `w_match = 1.0`, `w_feat = 0.05` (repo default), `w_cls = 0.5`, focal `γ = 2`.

---

## 6. Training recipe

Staged, to avoid destabilising the existing confidence-weighted regression (which assumes
all-positives):

1. **Stage A — warm start.** Load the released UniCorrn weights. Freeze backbones + fusion +
   decoder. Train **only** `intrinsics_encoder` + `frustum_head` with negatives. Confirms the
   representation already separates in/out (it should — `corr` in-bounds is a strong cue).
2. **Stage B — joint fine-tune.** Unfreeze decoder (and optionally fusion). Train all three tasks
   with masked losses + negatives. Keep regression strictly masked to positives.
3. **Stage C (optional) — distil light variant.** If only membership is needed at deployment,
   train a shallow classifier on fusion-encoder point features (skip the 8-layer decoder) toward
   the Stage-B frustum predictions, for speed.

---

## 7. Files to change (checklist)

- `unicorrn/model/embedder/` — add `IntrinsicsEncoder` (+ export).
- `unicorrn/model/modules/unified_query_decoder.py` — add `FrustumHead`, instantiate, extend
  `forward_pcd_to_img` / `forward_img_to_pcd` return signatures, accept `cam_params`.
- `unicorrn/model/unicorrn.py` — thread `cam_params`, surface `frustum_predictions`, allow dense
  queries.
- `unicorrn/datasets/img2pcd/*` — negative sampling, label + `valid_match` + `cam_params` emission.
- `unicorrn/trainer/unified_functions.py` — mask matching/conf/InfoNCE by `valid_match`; add
  `FrustumClassificationLoss`; new combined loss `GMFrustumMultiTaskLoss`.
- `unicorrn/trainer/functions.py` — register the new loss.
- `configs/models/*.yml` — `FRUSTUM_HEAD: True`, head dims, intrinsics dim.
- `configs/trainers/*.yml` — `frustum_wt`, focal `gamma`, negative ratio, staged freezing flags.

Respect repo rules: modules ≤ 160 lines, Google-style docstrings without arg/return types,
relative intra-package imports, no inline `# type: ignore`, `assert` only in tests.

---

## 8. Evaluation

- **Classification:** precision / recall / F1 / AP for in-frustum; **boundary recall** (points
  within a metric band of the cone surface) — the band that actually carries Z signal.
- **Calibration relevance:** Z-observability check — sweep `t_z`, plot `L_z(T)` curvature vs a
  pure reprojection cost; expect a sharper minimum in `t_z`.
- **No-regression check:** matching AUC / registration recall must not drop vs baseline after
  adding the auxiliary task.

---

## 9. Risks & mitigations

- **Negatives destabilise confidence regression** → mask strictly to positives; Stage-A warm start.
- **Class imbalance** → focal loss + balanced negative ratio.
- **Boundary ambiguity** → only boundary points carry Z signal; report boundary-band metrics, not
  just global accuracy.
- **Intrinsics leakage of extrinsic** → audit that `cam_params` contains **no** pose; keep labels
  the only place `T` is used.
- **Dense-query cost** → subsample or Stage-C light head.
- **Definition drift** → keep (a) strictly geometric (no z-buffer); if co-visibility is ever
  wanted, that is a separate label and a different head, not a config flag on this one.

---

## 12. Implementation as built (authoritative)

Branch `frustum-multitask` in fork `UniCorrn-Frustum`. The fork preserves the uncommitted RTX
3090 `curope` changes (`setup.py` `sm_86`, `kernels.cu` dispatch macro). All new code follows
`CLAUDE.md` (type hints, Google docstrings, relative imports, new modules ≤ 160 lines).

### 12.1 What it does

In the **pcd → image** direction, a third head classifies a *separate* set of 3D point queries
(balanced in/out-frustum) as inside/outside the camera viewing cone — alongside the unchanged
matching (`corr`) and confidence (`info`) heads. Inputs are point cloud + image only; `K, T` are
used solely to make labels offline.

### 12.2 Key design choices realised

- **No camera params as input** (§1.4). The head reads only the decoder's internal state.
- **Separate frustum query set**, not a unified masked set. Consequence: the existing matching /
  confidence / InfoNCE losses are **untouched and unmasked** (they still run on all-positive
  correspondence queries). This is lower-risk than the DETR-style masking originally sketched and
  reuses `decoder.forward_pcd_to_img` verbatim with a different query tensor.
- **Geometry-aware head** reusing `Mlp`: `logit = MLP([appearance q ; positional hidden_state ;
  predicted image xy])`, where the predicted xy is the per-layer attention soft-argmax (`gm`).
- **Deep supervision**: a logit per decoder layer, γ-decayed focal BCE (mirrors the matching
  auxiliary loss).

### 12.3 Files changed / added

New:
- `unicorrn/model/blocks/frustum.py` — `FrustumHead`.
- `unicorrn/datasets/img2pcd/frustum_labels.py` — `build_frustum_queries` (geometric label +
  balanced sampling + normalisation).

Edited:
- `unicorrn/model/blocks/__init__.py` — export `FrustumHead`.
- `unicorrn/model/modules/unified_query_decoder.py` — build `frustum_head`; collect per-layer
  `frustum_out` in `forward_pcd_to_img`; return it (7-tuple).
- `unicorrn/model/unicorrn.py` — `_frustum_pcd_to_img` helper; `frustum_query_pos_3d` kwarg on
  `forward_img_to_pcd`; surface `frustum_predictions` / `frustum_intermediates`; update the three
  `forward_pcd_to_img` call sites to the 7-tuple.
- `unicorrn/trainer/unified_functions.py` — `FrustumClassificationLoss` (registered).
- `unicorrn/trainer/unified_trainer.py` — `frustum_wt`; frustum step inside `_run_step_img2pcd`;
  thread `frustum_loss_fn` through `run_step`.
- `train.py` — build `frustum_loss_fn` from optional `cfg.FRUSTUM_LOSS`; pass to `run_step`.
- `unicorrn/datasets/img2pcd/sevenscenes_hard.py` — `return_frustum` / `num_frustum_queries`;
  emit `frustum_queries` + `frustum_labels`.
- `unicorrn/datasets/collate_functions.py` — stack frustum keys in
  `ImageToPointRegistrationCollateFn`.
- `configs/trainers/trainer_frustum_2d3d.yml` — new recipe: `img2pcd_wt: 1.0`, `frustum_wt: 1.0`,
  7Scenes `return_frustum: True`, `FRUSTUM_LOSS` block.

### 12.4 Backward compatibility

Frustum is fully gated: with no `FRUSTUM_LOSS` in the config and `return_frustum` unset, the model
builds an (unused) `frustum_head` but training/eval behave exactly as before. `safe_load_weights`
absorbs the extra head when loading released checkpoints.

### 12.5 How to run

```
accelerate launch train.py \
  --model_config configs/models/unicorrn_large_stage1.yml \
  --trainer_config configs/trainers/trainer_frustum_2d3d.yml \
  --output_dir ./output/frustum --training_stage stage_1
```

The `frustum_loss` scalar appears in the logged loss details. Recommended: warm-start from a
released UniCorrn checkpoint (`RESUME.CKPT_PATH`) and, for a first run, freeze the backbones so
only the matcher + frustum head adapt.

### 12.6 Not yet done / follow-ups

- Replicate the dataset change in `rgbdscenes.py` (and the joint collate
  `JointTrainingCollateFn._prepare_img2pcd`) if those loaders are used.
- Add a validation metric (precision/recall/F1, boundary-band recall) in
  `UnifiedTrainer.evaluate`.
- Downstream `L_z(T)` consistency term for calibration (Section 1.3) lives in the calibration
  codebase, not here.

