import gc
from typing import List, Literal, Optional, Type, Union

import torch
from diffusers import FluxPipeline
from diffusers.models.attention_processor import FluxAttnProcessor2_0

from dit_edit.core.flux_pipeline import EditedFluxPipeline
from dit_edit.core.hooks import locate_block
from dit_edit.core.processors import DitEditProcessor
from dit_edit.core.qkv_cache.cache import QKVCache


class QKVCacheFluxHandler:
    """Used to cache queries, keys and values of a FluxPipeline."""

    def __init__(
        self,
        pipe: Union[FluxPipeline, EditedFluxPipeline],
        positions_to_cache: List[str] = None,
        positions_to_cache_foreground: List[str] = None,
        inject_kv: Literal["image", "text", "both"] = None,
        text_seq_length: int = 512,
        q_mask: Optional[torch.Tensor] = None,
        processor_class: Optional[Type] = DitEditProcessor,
    ):
        if not isinstance(pipe, FluxPipeline) and not isinstance(
            pipe, EditedFluxPipeline
        ):
            raise NotImplementedError(f"QKVCache not yet implemented for {type(pipe)}.")

        self.pipe = pipe

        if positions_to_cache is not None:
            self.positions_to_cache = positions_to_cache
        else:
            # act on all transformer layers
            self.positions_to_cache = []

        if positions_to_cache_foreground is not None:
            self.positions_to_cache_foreground = positions_to_cache_foreground
        else:
            self.positions_to_cache_foreground = []

        self._cache = {"query": [], "key": [], "value": []}

        # Set Cached Processor to perform editing
        all_layers = [f"transformer.transformer_blocks.{i}" for i in range(19)] + [
            f"transformer.single_transformer_blocks.{i}" for i in range(38)
        ]
        for module_name in all_layers:
            inject_kv = "image" if module_name in self.positions_to_cache else None
            inject_kv_foreground = module_name in self.positions_to_cache_foreground

            module = locate_block(pipe, module_name)
            module.attn.set_processor(
                processor_class(
                    external_cache=self._cache,
                    inject_kv=inject_kv,
                    inject_kv_foreground=inject_kv_foreground,
                    text_seq_length=text_seq_length,
                    q_mask=q_mask,
                )
            )

    @property
    def cache(self) -> QKVCache:
        """Returns a dictionary initialized as {"query": [], "key": [], "value": []}.
            After calling a forward pass for pipe, queries, keys and values will be
            appended in the respective list for each layer.

        Returns:
        Dict[str, List[torch.Tensor]]: cache dictionary containing 'query', 'key' and 'value'
        """
        return self._cache

    def clear_cache(self) -> None:
        del self._cache
        gc.collect()  # force Python to clean up unreachable objects
        torch.cuda.empty_cache()  # tell PyTorch to release unused GPU memory from its cache
        self._cache = {"query": [], "key": [], "value": []}

        for module_name in self.positions_to_cache:
            module = locate_block(self.pipe, module_name)
            module.attn.set_processor(FluxAttnProcessor2_0())
