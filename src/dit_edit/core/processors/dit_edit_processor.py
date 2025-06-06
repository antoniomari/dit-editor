from typing import Literal, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from dit_edit.core.qkv_cache.cache import QKVCache


class DitEditProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        external_cache: QKVCache,
        inject_kv: Literal["image", "text", "both"] = None,
        inject_kv_foreground: bool = False,
        text_seq_length: int = 512,
        q_mask: Optional[torch.Tensor] = None,
        call_max_times=None,
        inject_q=True,
        inject_k=True,
        inject_v=True,
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
        self.inject_q = inject_q
        self.inject_k = inject_k
        self.inject_v = inject_v

        self.call_max_times = call_max_times
        if self.call_max_times is not None:
            self.num_calls = call_max_times
        else:
            self.num_calls = None
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

        # hidden states are the image patches (B, 4096, hidden_dim)

        # encoder_hidden_states are the text tokens (B, 512, hidden_dim)

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

            # concat inputs for attention -> (B, num_heads, 512 + 4096, head_dim)
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # TODO: try first without mask
        # Cache Q, K, V
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

        # Batch is formed as follow:
        # - background image (0)
        # - foreground image (1)
        # - composition(s) (2, 3, ...)
        # Create the combined attention mask, by forming Q_comp and K_comp, taking the Q and K of the background image
        # when outside of the mask, the one of the foreground image when inside the mask

        if self.num_calls is None or self.num_calls > 0:
            if self.inject_kv_foreground:
                if self.inject_k:
                    key[2:, :, start_idx:] = torch.where(
                        mask, key[1:2, :, start_idx:], key[0:1, :, start_idx:]
                    )
                if self.inject_q:
                    query[2:, :, start_idx:] = torch.where(
                        mask, query[1:2, :, start_idx:], query[0:1, :, start_idx:]
                    )
                if self.inject_v:
                    value[2:, :, start_idx:] = torch.where(
                        mask, value[1:2, :, start_idx:], value[0:1, :, start_idx:]
                    )
            else:
                if self.inject_k:
                    key[2:, :, start_idx:] = torch.where(
                        mask, key[2:, :, start_idx:], key[0:1, :, start_idx:]
                    )
                if self.inject_q:
                    query[2:, :, start_idx:] = torch.where(
                        mask, query[2:, :, start_idx:], query[0:1, :, start_idx:]
                    )
                if self.inject_v:
                    value[2:, :, start_idx:] = torch.where(
                        mask, value[2:, :, start_idx:], value[0:1, :, start_idx:]
                    )

            if self.num_calls is not None:
                self.num_calls -= 1

        # Use the combined attention map to compute attention using V from the composition image
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # hidden_states[2:, :, start_idx:] = torch.where(mask, weightage * hidden_states[1:2, :, start_idx:] + (1-weightage) * hidden_states[2:, :, start_idx:], hidden_states[2:, :, start_idx:])

        # concatenate the text
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
