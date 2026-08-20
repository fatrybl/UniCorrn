"""Flash-attention variant of the query matching decoder.

Structurally identical to :class:`QueryMatchingDecoder` -- same parameters, same frustum
head, same forward -- differing only in the attention kernel its blocks use. The
non-FA blocks concatenate the value streams into one xformers memory-efficient call; the
FA blocks issue separate ``gaussian_flash_attn`` calls, which run in bfloat16 and cannot
exceed a head dim of 256 after the Gaussian kernel's +8 padding. ``NUM_HEADS`` must
therefore be at least 2 at ``DEC_EMBED_DIM`` 512.

Inheriting rather than copying is what keeps the two in step: a change to the frustum
head or to a forward pass reaches both.
"""

from ..blocks import DualStreamQueryDecoderBlockFA
from .build import DECODER_REGISTRY
from .unified_query_decoder import QueryMatchingDecoder


@DECODER_REGISTRY.register()
class QueryMatchingDecoderFA(QueryMatchingDecoder):
    """Query matching decoder whose blocks attend with flash attention."""

    block_cls = DualStreamQueryDecoderBlockFA

    def block_kwargs(self, index):
        """Flag the first block, which attends on appearance only.

        Args:
            index: Block position in the stack.
        """
        return {"init": index == 0}


__all__ = ["QueryMatchingDecoderFA"]
