"""Flash-attention variants of the query matching decoder.

Structurally identical to :class:`QueryMatchingDecoder` -- same parameters, same frustum
head, same forward -- differing only in the attention kernel their blocks use. The
non-FA blocks concatenate the value streams into one xformers memory-efficient call; the
FA blocks issue separate ``gaussian_flash_attn`` calls, which run in bfloat16 and cannot
exceed a head dim of 256 after the Gaussian kernel's +8 padding. ``NUM_HEADS`` must
therefore be at least 2 at ``DEC_EMBED_DIM`` 512. The FA4 decoder keeps that head split
and runs the same attention through FlashAttention-4, the value streams merged again.

Inheriting rather than copying is what keeps the three in step: a change to the frustum
head or to a forward pass reaches all of them.
"""

from ..blocks import DualStreamQueryDecoderBlockFA, DualStreamQueryDecoderBlockFA4
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


@DECODER_REGISTRY.register()
class QueryMatchingDecoderFA4(QueryMatchingDecoderFA):
    """The flash-attention decoder with FlashAttention-4 blocks."""

    block_cls = DualStreamQueryDecoderBlockFA4


__all__ = ["QueryMatchingDecoderFA", "QueryMatchingDecoderFA4"]
