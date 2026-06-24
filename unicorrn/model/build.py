import torch

from ..utils import Registry
from ..utils.config import get_cfg

MODEL_REGISTRY = Registry("MODEL")


def build_model(name, cfg_path=None, cfg=None, default=False, weights_path=None):
    """
    Builds a model from a model name and config. Also supports loading weights

    Parameters
    ----------
    name : str
        Name of the model to build
    cfg_path : str, optional
        Path to a config file. If not provided, will use the default config
        for the model
    cfg : CfgNode object, optional
        Custom config object.
    weights_path : str, optional
        Path to a weights file

    Returns
    -------
    torch.nn.Module
        The model
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry.")

    if cfg is None:
        assert cfg_path is not None, "Please provide a config path."
        cfg = get_cfg(cfg_path)

    model = MODEL_REGISTRY.get(name)
    model = model(cfg)

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    return model
