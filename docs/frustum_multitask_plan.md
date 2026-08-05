# UniCorrn Multi-Task Extension — Matching + Confidence + Geometric Frustum Classification

Status: **implemented** on branch `frustum-multitask` (fork `UniCorrn-Frustum`).
Sections 1–2 give the theory; **Section 12 is the authoritative as-built description** and
**Section 13 records what training measured and which parts of the design it revises**.
Section 12 supersedes the earlier exploratory proposal (notably: the model takes **no camera parameters as
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

- **The signed-distance regression is not wired on this branch.** `build_frustum_queries`
  returns `(queries, labels)` only, and `UnifiedTrainer._run_step_img2pcd` calls
  `frustum_loss_fn(intermediates, labels)` with no `distances`, so `dist_weight` is inert and
  the head's second channel is trained only through the logit. Section 13 explains why this
  matters more than it looks. Emit the target from `frustum_labels.py` and pass it through.
- Replicate the dataset change in `rgbdscenes.py` (and the joint collate
  `JointTrainingCollateFn._prepare_img2pcd`) if those loaders are used. Note the intrinsics
  rescale in `sevenscenes_hard.py` / `rgbdscenes.py` scales `fx, cx` by the *height* ratio and
  `fy, cy` by the *width* ratio; the axes are swapped. It cancels for 7Scenes (640×480 →
  384×512, both 0.8) but will corrupt frustum labels on any non-aspect-preserving resize.
- Add a validation metric (precision/recall/F1, boundary-band recall) in
  `UnifiedTrainer.evaluate`.
- Downstream `L_z(T)` consistency term for calibration (Section 1.3) lives in the calibration
  codebase, not here.

---

## 13. Measured behaviour and the fixes it implies

Four epochs of the downstream VAPE fine-tune (nuScenes + S3DIS + S3E + ScanNet++, `dist_weight
= 0.5`, so the regression *is* active there) converged to `frustum_acc ≈ 0.91` with
`frustum_edge_acc ≈ 0.69` in the `|s*| < 0.02` band, both flat over the last 1000 steps. Three
findings generalise back to this design.

### 13.1 Accuracy is the regression's, not the classifier's

`logit = d · e^{log_scale}` with `e^{log_scale} > 0`, so `σ(logit) > ½ ⟺ d > 0`: the predicted
class is exactly `sign(d)`. The target satisfies `s* > 0 ⟺ in-frustum` by construction, so
**classification accuracy equals the sign agreement of the distance regression**. The BCE term
shapes gradients but never casts the vote, and the temperature cannot change a single
prediction. Anything aimed at accuracy has to act on `d`.

This is why the missing regression (§12.6) matters beyond a disabled term: without it, `d` is
fitted only through BCE, which determines it just up to a positive scale — it is a scaled
log-odds, not a distance — and the identity above then gives the classifier nothing extra.

### 13.2 `huber_beta` must match the boundary band — but not go below it

SmoothL1 has gradient `min(|e| / beta, 1)`. At the default `beta = 0.1` against a target
spanning `±0.5`, a boundary-scale error of `0.02` gets `0.2` gradient while a tail error of
`0.4` gets `1.0` — five times more weight on points whose class no decision depends on. Setting
`beta` to the band scale puts boundary errors in the linear regime at full gradient and leaves
the tails unchanged.

The failure mode on the other side is sharper than expected. Once `beta` drops below the error
the model can actually achieve, *every* residual is in the linear regime, the term is L1, and its
optimum is the conditional **median** rather than the mean. While the matcher is weak the
features are near-uninformative, the target's conditional median collapses to ≈ 0, and band
predictions land on zero with an arbitrary sign — accuracy pinned at chance. Measured at
`beta = 0.02`: band MAE reached 0.030, then *regressed* to 0.052 while band accuracy sat at 0.51.
Keep the library default of 0.1 until the band error is demonstrably below it.

### 13.3 Sample the band, and measure it

`build_frustum_queries` splits 50/50 *by class* and draws uniformly within each, which places
only a few percent of queries in the decision band — band membership scales with the cone's
surface, not its volume, while §1.2's axial signal exists only there. Drawing a fixed share of
each class from `|s*| < 0.05` raises the in-band share roughly 3× at unchanged class balance.

For diagnostics, pooled distance MAE is misleading: it is dominated by saturated targets and
moves largely independently of accuracy. Report MAE restricted to the band. For boundary error
`σ` and `|s*|` roughly uniform across an evaluation band `B`, expected band accuracy is
`Φ(a) + (φ(a) − φ(0))/a` with `a = B/σ` — at `B = 0.02`, `σ = 0.02` gives 0.68 and `σ = 0.0067`
gives 0.87, which is how to size the sharpening a target accuracy requires.

### 13.4 T-independence of the *inputs* is not T-independence of the *frame*

§1.4 keeps `T` out of the model's inputs so membership cannot collapse to a reprojection.
That is necessary but not sufficient. The label is a function of `T`; if the cloud reaches
the model in the sensor's own frame and the camera is rigidly mounted, membership becomes a
fixed function of the input coordinates and the head can satisfy the loss without reading
the image at all. `T` was never fed in — it leaked through the coordinate frame.

Measured on the downstream fixed-rig pool (nuScenes / S3DIS / S3E / ScanNet++): about nine
distinct extrinsics across the whole training set. A 2-layer MLP on the normalised
coordinates alone, no image, split by scene and scored on the same balanced query set the
head uses (chance 0.500), reaches **0.79–0.83** — against the full model's 0.911. Most of
the headline accuracy needs no image at all. Inside the boundary band the same probe sits
at **0.48–0.56**, i.e. chance, at every randomisation level, while the trained model
reaches 0.691: whatever the head genuinely learned from the image lives entirely in the
band, which is a second argument for §13.3's band metrics.

Three consequences for this branch:

- `normalize_coord` removes translation and scale exactly, so the **rotation** of the input
  frame is the only residual rig information and the only thing worth randomising.
- **But full `SO(3)` was not trainable here, and the probe does not predict that.** The
  probe measures what a small MLP extracts from coordinates, not what the full model can
  learn. At `max_rotation=180°` from a random decoder init the downstream model did not
  recover: `img2pcd_l1` plateaued at 0.55 against a 0.31 baseline, band accuracy stayed at
  chance for 899 steps. Full `SO(3)` makes the task complete 2D-3D registration with no
  pose prior — exactly what stage-1/stage-2 pretraining provides, and not something a
  random decoder bootstraps on 7 300 samples. `random_se3(max_rotation=45°)` removes only
  ~0.04–0.08 of the leak but is at least trainable. Randomising **yaw only** would break a
  fixed rig's memorised azimuth while preserving the gravity prior the point encoder
  depends on; that is the untested middle option worth trying first.
- `sevenscenes_hard.py` / `rgbdscenes.py` apply `random_sample_small_transform(scale=0.1)`
  — deliberate weak jitter, far too small to decorrelate a frame. That is acceptable for
  7Scenes and RGBD-V2, where a handheld camera makes `T` genuinely vary per frame, and
  **not** acceptable for any fixed-rig dataset added later. Measure the spread of `T`, and
  run the coords-only probe, before trusting a frustum number on a new loader.

### 13.5 Two limits this design has not yet tested

- **The temperature has no finite optimum.** Because the label is exactly `sign(s*)`, the data
  is separable in `d` and BCE drives `log_scale` up without bound; measured, it rises linearly
  in the step count (`R² = 0.99998`) with weight decay ~230× too weak to oppose it. Harmless for
  accuracy (§13.1), fatal for treating `σ(d/T)` as calibrated over a long run.
- **§1.4's implicit-`K` claim is untested here.** With a single IMG2PCD dataset at one fixed
  intrinsic, "infers `K` from image content" and "memorised 7Scenes' `K`" are
  indistinguishable. Adding RGBD-V2 turns the assumption into a measurement.

