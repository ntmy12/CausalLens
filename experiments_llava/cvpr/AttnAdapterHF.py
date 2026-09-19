import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads along head dimension (for Grouped Query Attention)
    (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None, unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key states."""
    if cos.dim() == 2 and position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    elif cos.dim() == 4:
        pass  # already shaped properly

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class AttnAdapterHF(nn.Module):
    """
    CausalLens Attention Adapter compatible with HuggingFace transformers >= 4.45.
    Designed for LLaVA-1.5 language model layers (LlamaDecoderLayer).
    
    Implements:
    - Path decomposition: sys, vis, rest tokens
    - Head-level Visual Causal Sensitivity score (s_score)
    - Dynamic Causal Gate (gamma_dynamic)
    - Head-level hybrid intervention: h_head = (1-gamma)*h_orig + gamma*[h_lang + lambda*s*(h_vis - h_lang)]
    - Post-O projection causal residual: attn_output += lambda * s_global * o_proj(delta_head)
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        lambda_causal: float = 0.15,
        gamma_mix: float = 0.15,
        sys_len: int = 35,
        img_len: int = 576,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.lambda_causal = float(lambda_causal)
        self.gamma_mix = float(gamma_mix)
        self.sys_len = int(sys_len)
        self.img_len = int(img_len)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Optional rotary embedding reference if needed
        self.rotary_emb = None

    def update_token_range(self, sys_len: int, img_len: int):
        """Update system and image token lengths per sample."""
        self.sys_len = int(sys_len)
        self.img_len = int(img_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Union[Cache, Tuple[torch.Tensor]]]]:

        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # 1) Q/K/V Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2) RoPE Position Embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            elif isinstance(past_key_value, tuple):
                kv_seq_len += past_key_value[0].shape[-2]

        if position_embeddings is None:
            if self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            else:
                raise ValueError("Rotary embedding or position_embeddings must be provided to AttnAdapterHF")
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 3) Update KV Cache
        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            elif isinstance(past_key_value, tuple):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                past_key_value = (key_states, value_states) if use_cache else None

        # 4) Repeat KV for GQA / MHA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 5) Attention Scores & Softmax
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # Handle causal mask slicing
            causal_mask = attention_mask[:, :, :, :attn_scores.shape[-1]]
            attn_scores = attn_scores + causal_mask

        # Fix precision issues in float16 inference on Turing T4
        if query_states.dtype == torch.float16:
            attn_scores = torch.where(torch.isinf(attn_scores), torch.zeros_like(attn_scores), attn_scores)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 6) Path Decomposition: sys, vis, rest
        curr_kv_len = value_states.shape[-2]
        sys_len = min(self.sys_len, curr_kv_len)
        img_len = self.img_len
        vis_start = sys_len
        vis_end = min(sys_len + img_len, curr_kv_len)

        A_lang = attn_weights[..., :vis_start]
        A_vis = attn_weights[..., vis_start:vis_end]
        A_rest = attn_weights[..., vis_end:]

        V_lang = value_states[..., :vis_start, :]
        V_vis = value_states[..., vis_start:vis_end, :]
        V_rest = value_states[..., vis_end:, :]

        if vis_start > 0:
            h_lang = torch.matmul(A_lang, V_lang)
        else:
            h_lang = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, device=device, dtype=query_states.dtype)

        if vis_end > vis_start:
            h_vis = torch.matmul(A_vis, V_vis)
        else:
            h_vis = torch.zeros_like(h_lang)

        if curr_kv_len > vis_end:
            h_rest = torch.matmul(A_rest, V_rest)
        else:
            h_rest = torch.zeros_like(h_lang)

        h_orig = h_lang + h_vis + h_rest

        # 7) Visual Causal Sensitivity Score
        if vis_end > vis_start:
            var = A_vis.var(dim=-1, keepdim=True)
            mean = A_vis.mean(dim=-1, keepdim=True)
            s_score = var / (mean + 1e-6)
            s_score = s_score / (s_score.mean(dim=1, keepdim=True) + 1e-6)
        else:
            s_score = torch.zeros(bsz, self.num_heads, q_len, 1, device=device, dtype=query_states.dtype)

        # 8) Dynamic Causal Gate (gamma)
        E_vis = (h_vis ** 2).mean()
        E_lang = (h_lang ** 2).mean()
        gamma_dynamic = E_lang / (E_vis + E_lang + 1e-6)
        gamma_dynamic = gamma_dynamic.view(1, 1, 1, 1)

        gamma = gamma_dynamic

        # 9) Head-level Hybrid Intervention
        h_head = (1.0 - gamma) * h_orig + gamma * (
            h_lang + self.lambda_causal * s_score * (h_vis - h_lang)
        )

        # 10) Output Projection
        h_head_flat = h_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(h_head_flat)

        # 11) Post-O Proj Causal Residual
        s_global = s_score.mean(dim=1, keepdim=False)
        delta_head = h_vis - h_lang
        delta_flat = delta_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        delta_proj = self.o_proj(delta_flat)

        attn_output = attn_output + self.lambda_causal * s_global * delta_proj

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
