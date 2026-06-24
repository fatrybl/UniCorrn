import torch.nn as nn

from ...utils import Registry

FUNCTIONAL_REGISTRY = Registry("FUNCTIONAL")


def get_functional(name, **kwargs):
    """
    Retrieve a component from the functional registry

    Parameters
    ----------
    name : str
        Name of the component
    kwargs : dict
        Additional keyword arguments
    """

    return FUNCTIONAL_REGISTRY.get(name)(**kwargs)


FUNCTIONAL_REGISTRY.register(obj=nn.GELU, name="gelu")
FUNCTIONAL_REGISTRY.register(obj=nn.LayerNorm, name="layer_norm")
