"""Names of the attention kernels a module can be built with, the config key that selects
one, and the check every module runs on the name it was given.

Not every module offers every kernel: LitePT and CroCo choose between ``flash`` and
``fa4``, the fusion blocks between ``xformers`` and ``fa4``, PTv3 between what its
``enable_flash`` flag selects (``None``) and ``fa4``.
"""

from . import fa4

ATTN_KERNEL_KEY = "ATTN_KERNEL"
FA4 = "fa4"
FLASH = "flash"
XFORMERS = "xformers"


def check_kernel(
    kernel: str | None, allowed: tuple[str | None, ...], module: str, attn_drop: float
) -> None:
    """Reject a kernel ``module`` has no path for; make sure FA4 can run when it is asked for.

    Args:
        kernel: Requested kernel name.
        allowed: The names the module implements.
        module: Name of the module being built, for the message.
        attn_drop: Its attention dropout, which FA4 cannot apply.

    Raises:
        ValueError: If ``kernel`` is not in ``allowed``.
    """
    if kernel not in allowed:
        raise ValueError(f"{module}: unknown attention kernel {kernel!r}; expected one of {allowed}")
    if kernel == FA4:
        fa4.configure(module, attn_drop)
