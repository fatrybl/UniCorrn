# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from .dust3r.datasets.arkitscenes import ARKitScenes as DUSt3R_ARKitScenes  # noqa
from .dust3r.datasets.blendedmvs import BlendedMVS as DUSt3R_BlendedMVS  # noqa
from .dust3r.datasets.co3d import Co3d as DUSt3R_Co3d  # noqa
from .dust3r.datasets.habitat import Habitat as DUSt3R_Habitat
from .dust3r.datasets.megadepth import MegaDepth as DUSt3R_MegaDepth  # noqa
from .dust3r.datasets.scannet import ScanNetDataset
from .dust3r.datasets.scannetpp import ScanNetpp as DUSt3R_ScanNetpp  # noqa
from .dust3r.datasets.staticthings3d import (  # noqa
    StaticThings3D as DUSt3R_StaticThings3D,
)
from .dust3r.datasets.waymo import Waymo as DUSt3R_Waymo  # noqa
from .dust3r.datasets.wildrgbd import WildRGBD as DUSt3R_WildRGBD  # noqa
from .mast3r_base_stereo_view_dataset import MASt3RBaseStereoViewDataset
from .unified_base_dataset import UnifiedBaseDataset


class ARKitScenes(DUSt3R_ARKitScenes, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class BlendedMVS(DUSt3R_BlendedMVS, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, split=None, **kwargs):
        super().__init__(*args, ROOT=ROOT, split=split, **kwargs)
        self.is_metric_scale = False


class Co3d(DUSt3R_Co3d, MASt3RBaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class MegaDepth(DUSt3R_MegaDepth, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class Habitat(DUSt3R_Habitat, MASt3RBaseStereoViewDataset):
    def __init__(self, size, *args, split, ROOT, **kwargs):
        super().__init__(size, *args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class ScanNetpp(DUSt3R_ScanNetpp, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class StaticThings3D(DUSt3R_StaticThings3D, MASt3RBaseStereoViewDataset):
    def __init__(self, ROOT, *args, mask_bg="rand", **kwargs):
        super().__init__(ROOT, *args, mask_bg=mask_bg, **kwargs)
        self.is_metric_scale = False


class Waymo(DUSt3R_Waymo, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class WildRGBD(DUSt3R_WildRGBD, MASt3RBaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class ScanNet(ScanNetDataset, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)


#####################################################################
#                Unified Datasets for Joint Training                #
#####################################################################


class MegaDepth_UnifiedDataset(DUSt3R_MegaDepth, UnifiedBaseDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class ScanNetpp_UnifiedDataset(DUSt3R_ScanNetpp, UnifiedBaseDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class ARKitScenes_UnifiedDataset(DUSt3R_ARKitScenes, UnifiedBaseDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class BlendedMVS_UnifiedDataset(DUSt3R_BlendedMVS, UnifiedBaseDataset):
    def __init__(self, *args, ROOT, split=None, **kwargs):
        super().__init__(*args, ROOT=ROOT, split=split, **kwargs)
        self.is_metric_scale = False


class Co3d_UnifiedDataset(DUSt3R_Co3d, UnifiedBaseDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class StaticThings3D_UnifiedDataset(DUSt3R_StaticThings3D, UnifiedBaseDataset):
    def __init__(self, ROOT, *args, mask_bg="rand", **kwargs):
        super().__init__(ROOT, *args, mask_bg=mask_bg, **kwargs)
        self.is_metric_scale = False


class ScanNet_UnifiedDS_V1(ScanNetDataset, UnifiedBaseDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)


class Waymo_UnifiedDataset(DUSt3R_Waymo, UnifiedBaseDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)


class WildRGBD_UnifiedDataset(DUSt3R_WildRGBD, UnifiedBaseDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
