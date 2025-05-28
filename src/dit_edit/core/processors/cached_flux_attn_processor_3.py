from typing import Literal, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from dit_edit.core.qkv_cache.cache import QKVCache


class CachedFluxAttnProcessor3_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        external_cache: QKVCache,
        inject_kv: Literal["image", "text", "both"] = None,
        inject_kv_foreground: bool = False,
        text_seq_length: int = 512,
        q_mask: Optional[torch.Tensor] = None,
    ):
        """Constructor for Cached attention processor.

        Args:
            external_cache (QKVCache): cache to store/inject values.
            inject_kv (Literal[&quot;image&quot;, &quot;text&quot;, &quot;both&quot;], optional): whether to inject image, text or both streams KV.
                If None, it does not perform injection but the full cache is stored. Defaults to None.
        """

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.cache = external_cache
        self.inject_kv = inject_kv
        self.inject_kv_foreground = inject_kv_foreground
        self.text_seq_length = text_seq_length
        self.q_mask = q_mask
        assert all(
            (cache_key in external_cache) for cache_key in {"query", "key", "value"}
        ), "Cache has to contain 'query', 'key' and 'value' keys."

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

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
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # # Cache Q, K, V
        # if self.inject_kv == "image":
        #     # NOTE: I am replacing key and values only for the image branch
        #     # NOTE: in default settings, encoder_hidden_states_key_proh.shape[2] == 512
        #     # the first element of the batch is the image whose key and value will be injected into all the other images
        #     key[1:, :, self.text_seq_length:] = key[:1, :, self.text_seq_length:]
        #     value[1:, :, self.text_seq_length:] = value[:1, :, self.text_seq_length:]
        # elif self.inject_kv == "text":
        #     key[1:, :, :self.text_seq_length] = key[:1, :, :self.text_seq_length]
        #     value[1:, :, :self.text_seq_length] = value[:1, :, :self.text_seq_length]
        # elif self.inject_kv == "both":
        #     key[1:] = key[:1]
        #     value[1:] = value[:1]
        # else: # Don't inject, store cache!
        #     self.cache["query"].append(query)
        #     self.cache["key"].append(key)
        #     self.cache["value"].append(value)

        # extend the mask to match key and values dimension:
        # Shape of mask is: (num_image_tokens, 1)
        mask = (
            self.q_mask.permute(1, 0).unsqueeze(0).unsqueeze(-1)
        )  # Shape: (1, num_image_tokens, 1, 1)
        # put mask on gpu
        mask = mask.to(key.device)
        # first check that we inject only kv in images:
        if self.inject_kv is not None and self.inject_kv != "image":
            raise NotImplementedError("Injecting is implemented only for images.")
        # the second element of the batch is the number of heads
        # The first element of the batch represents the background image, the second element of the batch
        # represents the foreground image. The third element represents the image where we want to inject
        # the key and value of the background image and foreground image according to the query mask.
        # Inject from background (element 0) where mask is True

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Get the index range after the text tokens
        start_idx = self.text_seq_length

        if self.inject_kv_foreground and self.inject_kv == "image":
            key[2:, :, start_idx:] = torch.where(
                mask, key[1:2, :, start_idx:], key[:1, :, start_idx:]
            )
            value[2:, :, start_idx:] = torch.where(
                mask, value[1:2, :, start_idx:], value[:1, :, start_idx:]
            )
        elif self.inject_kv == "image" and not self.inject_kv_foreground:
            key[2:, :, start_idx:] = torch.where(
                mask, key[2:, :, start_idx:], key[:1, :, start_idx:]
            )
            value[2:, :, start_idx:] = torch.where(
                mask, value[2:, :, start_idx:], value[:1, :, start_idx:]
            )
        elif self.inject_kv is None and self.inject_kv_foreground:
            key[2:, :, start_idx:] = torch.where(
                mask, key[1:2, :, start_idx:], key[2:, :, start_idx:]
            )
            value[2:, :, start_idx:] = torch.where(
                mask, value[1:2, :, start_idx:], value[2:, :, start_idx:]
            )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # mask hidden states from bg:
        # hidden_states = hidden_states_fg[:, :, start_idx:] * mask + hidden_states_bg[:, :, start_idx:] * (~mask)

        # concatenate the text
        # hidden_states = torch.cat([hidden_states_bg[:, :, :start_idx], hidden_states], dim=2)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
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
