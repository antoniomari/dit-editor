# Add parent directory to sys.path
from collections import defaultdict
import gc
import os, sys
from pathlib import Path
parent_dir = Path.cwd().parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from typing import Dict, List, Literal, Optional, TypedDict
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from cache_and_edit.hooks import locate_block
import torch.nn.functional as F


class QKVCache(TypedDict):
    query: List[torch.Tensor]
    key: List[torch.Tensor]
    value: List[torch.Tensor]


class CachedFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, external_cache: QKVCache, inject_kv: Literal["image", "text", "both"]= None):
        """Constructor for Cached attention processor.

        Args:
            external_cache (QKVCache): cache to store/inject values.
            inject_kv (Literal[&quot;image&quot;, &quot;text&quot;, &quot;both&quot;], optional): whether to inject image, text or both streams KV. 
                If None, it does not perform injection but the full cache is stored. Defaults to None.
        """

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.cache = external_cache
        self.inject_kv = inject_kv
        assert all((cache_key in external_cache) for cache_key in {"query", "key", "value"}), "Cache has to contain 'query', 'key' and 'value' keys."

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Cache Q, K, V
        if self.inject_kv == "image":
            # NOTE: I am replacing key and values only for the image branch
            # NOTE: in default settings, encoder_hidden_states_key_proh.shape[2] == 512
            key[:, :, encoder_hidden_states_key_proj.shape[2]:] = self.cache["key"].pop(0)[:, :, encoder_hidden_states_key_proj.shape[2]:]
            value[:, :, encoder_hidden_states_value_proj.shape[2]:] = self.cache["value"].pop(0)[:, :, encoder_hidden_states_value_proj.shape[2]:]
        elif self.inject_kv == "text":
            key[:, :, :encoder_hidden_states_key_proj.shape[2]] = self.cache["key"].pop(0)[:, :, :encoder_hidden_states_key_proj.shape[2]] 
            value[:, :, :encoder_hidden_states_value_proj.shape[2]] = self.cache["value"].pop(0)[:, :, :encoder_hidden_states_value_proj.shape[2]]
        elif self.inject_kv == "both":
            key = self.cache["key"].pop(0)
            value = self.cache["value"].pop(0)
        else: # Don't inject, store cache!
            self.cache["query"].append(query)
            self.cache["key"].append(key)
            self.cache["value"].append(value)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class QKVCacheFluxHandler:
    """Used to cache queries, keys and values of a FluxPipeline.
    """

    def __init__(self, pipe: FluxPipeline, 
                 positions_to_cache: List[str] = None,
                 inject_kv: Literal["image", "text", "both"] = None,
                 cache_to_inject: QKVCache = None):
        
        if cache_to_inject is not None:
            self._cache = cache_to_inject
        else:
            self._cache = {"query": [], "key": [], "value": []}

        if positions_to_cache is not None:
            # TODO: extens for other models
            if not isinstance(pipe, FluxPipeline):
                raise NotImplementedError(f"QKVCache not yet implemented for {type(pipe)}.")
            transformer: FluxTransformer2DModel = pipe.transformer
            for layer in transformer.transformer_blocks:
                layer.attn.set_processor(CachedFluxAttnProcessor2_0(external_cache=self._cache, inject_kv=inject_kv))
            for layer in transformer.single_transformer_blocks:
                layer.attn.set_processor(CachedFluxAttnProcessor2_0(external_cache=self._cache, inject_kv=inject_kv))
        else:

            for module_name in positions_to_cache:
                module = locate_block(module_name)
                module.attn.set_processor(CachedFluxAttnProcessor2_0(external_cache=self._cache, inject_kv=inject_kv))


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
        # TODO: check if we have to force clean GPU memory
        del(self._cache)
        gc.collect()              # force Python to clean up unreachable objects
        torch.cuda.empty_cache()  # tell PyTorch to release unused GPU memory from its cache
        self._cache = {"query": [], "key": [], "value": []}



