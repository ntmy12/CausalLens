import os
import sys
import json
import math
import argparse
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, set_seed
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from transformers.cache_utils import Cache
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
from tqdm import tqdm

# Ensure module import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causallens_utils.coco_path_finder import find_coco_val2014_dir
from pope_evaluator import evaluate_pope, extract_prediction


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads (for GQA)
    (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttnAdapter(nn.Module):
    """
    Custom Attention Adapter for Qwen2-VL
    
    Implements:
    - Head-level hybrid intervention
    - Enhancement of attention to image tokens
    - Suppression of attention to system/text tokens
    
    Key hyperparameters:
    - lambda_causal: intervention strength (higher = stronger modification)
    - gamma_mix: mixing ratio between residual and replacement
    - sys_len: number of system prompt tokens (before image tokens)
    - img_len: number of image tokens
    """
    
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        lambda_causal: float = 0.15,
        gamma_mix: float = 0.15,
        sys_len: int = 31,
        img_len: int = 256,
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Hyperparameters for attention modification
        self.lambda_causal = float(lambda_causal)
        self.gamma_mix = float(gamma_mix)
        self.sys_len = int(sys_len)
        self.img_len = int(img_len)
        
        # Model dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_scaling = config.rope_scaling
        
        # Q/K/V/O projections (will be loaded from original attention)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embedding (will be copied from original)
        self.rotary_emb = None
        
        # Storage for attention weights (for analysis)
        self.last_attn_weights = None
        self.last_hidden_states = None
        self.prefill_attn_weights = None
    
    def update_token_range(self, sys_len: int, img_len: int):
        """Update token range for each sample"""
        self.sys_len = int(sys_len)
        self.img_len = int(img_len)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Custom forward with attention modification.
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device
        
        # Store hidden states for analysis
        self.last_hidden_states = hidden_states.detach().clone()
        
        # ========== Step 1: Q/K/V projections ==========
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # ========== Step 2: Apply Rotary Position Embedding ==========
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        
        # ========== Step 3: Handle KV Cache ==========
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # ========== Step 4: Repeat K/V for GQA ==========
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        kv_seq_len = key_states.shape[2]
        
        # ========== Step 5: Compute Attention Scores ==========
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_scores = attn_scores + causal_mask
        
        # Fix precision issues in float16 inference
        if query_states.dtype == torch.float16:
            attn_scores = torch.where(torch.isinf(attn_scores), torch.zeros_like(attn_scores), attn_scores)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Store attention weights for analysis
        self.last_attn_weights = attn_weights.detach().clone()
        if q_len > 1:
            self.prefill_attn_weights = attn_weights.detach().clone()
        
        # ========== Step 6: Path Decomposition & Causal Intervention ==========
        # Split into image tokens and non-image tokens (lang = sys + rest)
        SYS_LEN = min(self.sys_len, kv_seq_len)
        IMG_LEN = self.img_len
        vis_start = SYS_LEN
        vis_end = min(SYS_LEN + IMG_LEN, kv_seq_len)
        
        # Safe slices - combine sys tokens and rest tokens as non-image tokens
        A_sys = attn_weights[..., :vis_start]
        A_vis = attn_weights[..., vis_start:vis_end]
        A_rest = attn_weights[..., vis_end:]
        
        V_sys = value_states[..., :vis_start, :]
        V_vis = value_states[..., vis_start:vis_end, :]
        V_rest = value_states[..., vis_end:, :]
        
        # Compute per-path head outputs
        h_sys = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, device=device, dtype=query_states.dtype)
        if vis_start > 0:
            h_sys = torch.matmul(A_sys, V_sys)
        
        h_rest = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, device=device, dtype=query_states.dtype)
        if kv_seq_len > vis_end:
            h_rest = torch.matmul(A_rest, V_rest)
        
        # Combine sys and rest as h_lang (non-image tokens)
        h_lang = h_sys + h_rest
        
        if vis_end > vis_start:
            h_vis = torch.matmul(A_vis, V_vis)
        else:
            h_vis = torch.zeros_like(h_lang)
        
        # Original full attention output
        h_orig = h_lang + h_vis
        
        # ========== Step 7: Compute Visual Causal Sensitivity Score ==========
        if vis_end > vis_start:
            var = A_vis.var(dim=-1, keepdim=True)
            mean = A_vis.mean(dim=-1, keepdim=True)
            s_score = var / (mean + 1e-6)
            s_score = s_score / (s_score.mean(dim=1, keepdim=True) + 1e-6)
        else:
            s_score = torch.zeros(bsz, self.num_heads, q_len, 1, device=device, dtype=query_states.dtype)
        
        # ========== Step 8: Dynamic Gamma Calculation ==========
        E_vis = (h_vis ** 2).mean()
        E_lang = (h_lang ** 2).mean()
        gamma_dynamic = E_lang / (E_vis + E_lang + 1e-6)
        gamma_dynamic = gamma_dynamic.view(1, 1, 1, 1)
        
        gamma = gamma_dynamic
        
        # ========== Step 9: Hybrid Intervention ==========
        h_head = (1.0 - gamma) * h_orig + gamma * (
            h_lang + self.lambda_causal * s_score * (h_vis - h_lang)
        )
        
        # ========== Step 10: Project Output ==========
        h_head_flat = h_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(h_head_flat)
        
        # ========== Step 11: Post O-proj Causal Residual ==========
        s_global = s_score.mean(dim=1, keepdim=False)
        delta_head = h_vis - h_lang
        delta_flat = delta_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        delta_proj = self.o_proj(delta_flat)
        
        # Final output with causal correction
        attn_output = attn_output + self.lambda_causal * s_global * delta_proj
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights


def find_vision_token_range(input_ids: torch.Tensor) -> Tuple[int, int]:
    """
    Find vision token range from input_ids for Qwen2-VL.
    vision_start_token_id: 151652 (<|vision_start|>)
    vision_end_token_id: 151653 (<|vision_end|>)
    """
    vision_start_id = 151652
    vision_end_id = 151653
    
    input_ids_flat = input_ids[0].tolist()
    
    vision_start_idx = None
    vision_end_idx = None
    
    for i, token_id in enumerate(input_ids_flat):
        if token_id == vision_start_id and vision_start_idx is None:
            vision_start_idx = i
        if token_id == vision_end_id:
            vision_end_idx = i
            break
    
    if vision_start_idx is None:
        vision_start_idx = 31
    if vision_end_idx is None:
        vision_end_idx = vision_start_idx + 258
    
    return vision_start_idx, vision_end_idx


def replace_attention_with_adapter(
    model,
    target_layer_range: Tuple[int, int] = (10, 20),
    lambda_causal: float = 0.15,
    gamma_mix: float = 0.15,
    sys_len: int = 31,
    img_len: int = 256,
) -> List[Qwen2VLAttnAdapter]:
    """
    Replace self_attn in specified layers [layer_start, layer_end] (inclusive) with Qwen2VLAttnAdapter.
    """
    adapters = []
    qwen2vl_model = model.model
    
    if hasattr(qwen2vl_model, 'language_model'):
        lm_model = qwen2vl_model.language_model
        layers = lm_model.layers
    elif hasattr(qwen2vl_model, 'layers'):
        layers = qwen2vl_model.layers
    else:
        raise AttributeError(f"Cannot find layers in model. Available attributes: {[n for n, _ in qwen2vl_model.named_children()]}")
    
    for i, layer in enumerate(layers):
        if target_layer_range[0] <= i <= target_layer_range[1]:
            print(f"[CausalLens Qwen2-VL] Replacing layer {i} self_attn with Qwen2VLAttnAdapter")
            
            attn_adapter = Qwen2VLAttnAdapter(
                config=model.config,
                layer_idx=i,
                lambda_causal=lambda_causal,
                gamma_mix=gamma_mix,
                sys_len=sys_len,
                img_len=img_len,
            )
            
            # Copy weights from original attention
            attn_adapter.q_proj.load_state_dict(layer.self_attn.q_proj.state_dict())
            attn_adapter.k_proj.load_state_dict(layer.self_attn.k_proj.state_dict())
            attn_adapter.v_proj.load_state_dict(layer.self_attn.v_proj.state_dict())
            attn_adapter.o_proj.load_state_dict(layer.self_attn.o_proj.state_dict())
            
            # Copy rotary embedding if exists
            if hasattr(layer.self_attn, 'rotary_emb'):
                attn_adapter.rotary_emb = layer.self_attn.rotary_emb
            
            # Preserve accelerate multi-GPU hook if present
            if hasattr(layer.self_attn, '_hf_hook'):
                attn_adapter._hf_hook = layer.self_attn._hf_hook
            
            # Convert to same dtype and device as original layer
            param = next(layer.self_attn.parameters())
            attn_adapter = attn_adapter.to(dtype=param.dtype, device=param.device)
            
            # Replace self_attn
            layer.self_attn = attn_adapter
            adapters.append(attn_adapter)
    
    print(f"[CausalLens Qwen2-VL] Successfully injected {len(adapters)} Qwen2VLAttnAdapter modules.")
    return adapters


def update_adapters_token_range(adapters: List[Qwen2VLAttnAdapter], sys_len: int, img_len: int):
    """Update token range for all adapters per sample."""
    for adapter in adapters:
        adapter.update_token_range(sys_len, img_len)


def run_qwen2vl_pope(
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    pope_dir: str = "experiments/data/POPE/coco",
    image_dir: Optional[str] = None,
    output_dir: str = "results",
    split: str = "all",
    lambda_causal: float = 0.15,
    gamma_mix: float = 0.15,
    layer_start: int = 10,
    layer_end: int = 20,
    max_new_tokens: int = 6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run POPE evaluation with CausalLens intervention on Qwen2-VL-7B.
    """
    set_seed(seed)
    
    if process_vision_info is None:
        raise ImportError(
            "qwen_vl_utils is required for running Qwen2-VL. "
            "Please install it with: pip install qwen_vl_utils"
        )
    
    # 1. Resolve image and annotation paths
    image_dir = find_coco_val2014_dir(image_dir)
    print(f"[Qwen2-VL Runner] COCO val2014 directory: {image_dir}")
    
    splits = ["random", "popular", "adversarial"] if split == "all" else [split]
    for s in splits:
        pope_file = os.path.join(pope_dir, f"coco_pope_{s}.json")
        if not os.path.exists(pope_file):
            raise FileNotFoundError(f"POPE annotation file not found: {pope_file}")
            
    # 2. Precision & Model Loading
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[Qwen2-VL Runner] Loading {model_path} with dtype={torch_dtype}, eager attention, device_map='auto'...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Restrict max_pixels to 313600 per HANDOVER requirements (speeds up generation)
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=313600,
    )
    
    # 3. Inject CausalLens Adapters
    adapters = replace_attention_with_adapter(
        model,
        target_layer_range=(layer_start, layer_end),
        lambda_causal=lambda_causal,
        gamma_mix=gamma_mix,
        sys_len=31,
        img_len=256,
    )
    
    all_results = {}
    
    # 4. Process splits
    for s in splits:
        pope_file = os.path.join(pope_dir, f"coco_pope_{s}.json")
        split_out_dir = os.path.join(output_dir, s)
        os.makedirs(split_out_dir, exist_ok=True)
        raw_output_path = os.path.join(split_out_dir, "raw_outputs.jsonl")
        metrics_output_path = os.path.join(split_out_dir, "metrics.json")
        config_output_path = os.path.join(split_out_dir, "run_config.json")
        
        run_config = {
            "model": "qwen2vl",
            "model_path": model_path,
            "split": s,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "prompt_suffix": "Please answer with yes or no.",
            "lambda_causal": lambda_causal,
            "gamma_mix": gamma_mix,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "seed": seed,
            "torch_dtype": str(torch_dtype),
            "attn_implementation": "eager",
        }
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
            
        with open(pope_file, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
            
        # Check already finished questions for resumability
        finished_qids = set()
        if os.path.exists(raw_output_path):
            with open(raw_output_path, "r", encoding="utf-8") as f_prev:
                for line in f_prev:
                    line = line.strip()
                    if line:
                        try:
                            finished_qids.add(json.loads(line).get("question_id"))
                        except Exception:
                            pass

        print(f"\n[Qwen2-VL Runner] Processing split: '{s}' ({len(samples)} samples, {len(finished_qids)} already done) -> {raw_output_path}")

        cached_image = None
        cached_image_path = None

        with open(raw_output_path, "a", encoding="utf-8") as f_out:
            for item in tqdm(samples, desc=f"Qwen2-VL POPE ({s})"):
                question_id = item["question_id"]
                if question_id in finished_qids:
                    continue
                image_name = item["image"]
                question = item["text"]
                label = item["label"]

                image_path = os.path.join(image_dir, image_name)
                if not os.path.exists(image_path):
                    continue
                
                # Include system preamble for sys_len and MANDATORY suffix for Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": f"{question} Please answer with yes or no."},
                        ],
                    }
                ]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move inputs to model device
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                
                # Find vision token range and update adapters
                vis_start, vis_end = find_vision_token_range(inputs["input_ids"])
                img_len = vis_end - vis_start
                sys_len = vis_start
                update_adapters_token_range(adapters, sys_len, img_len)
                
                # Strict Greedy Generation (max_new_tokens=6)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                
                pred_decision = extract_prediction(output_text)
                
                result = {
                    "question_id": question_id,
                    "image": image_name,
                    "question": question,
                    "label": label,
                    "pred": pred_decision,
                    "answer": output_text,
                    "text": output_text,
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

                # Clean cache periodically to avoid VRAM fragmentation across 3000 questions
                if torch.cuda.is_available() and (question_id % 500 == 0):
                    torch.cuda.empty_cache()
                
        # Evaluate split
        metrics = evaluate_pope(pope_file, raw_output_path, metrics_output_path)
        all_results[s] = {
            "metrics": metrics,
            "raw_output": raw_output_path,
            "config": run_config,
        }
        print(f"[Qwen2-VL Runner] Split '{s}' Results:")
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall   : {metrics['recall']*100:.2f}%")
        print(f"  F1-Score : {metrics['f1']*100:.2f}%")
        print(f"  Yes-Ratio: {metrics['yes_ratio']*100:.2f}%")
        
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL Attention Adapter for POPE Evaluation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--pope_dir", type=str, default="experiments/data/POPE/coco")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/qwen2vl_pope")
    parser.add_argument("--split", type=str, default="all", choices=["random", "popular", "adversarial", "all"])
    parser.add_argument("--lambda_causal", type=float, default=0.15)
    parser.add_argument("--gamma_mix", type=float, default=0.15)
    parser.add_argument("--layer_start", type=int, default=10)
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_qwen2vl_pope(
        model_path=args.model_path,
        pope_dir=args.pope_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        split=args.split,
        lambda_causal=args.lambda_causal,
        gamma_mix=args.gamma_mix,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
