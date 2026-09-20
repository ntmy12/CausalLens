import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from PIL import Image
import torch
from transformers import set_seed, AutoProcessor, LlavaForConditionalGeneration

# Ensure module import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causallens_utils.coco_path_finder import find_coco_val2014_dir
from experiments_llava.cvpr.AttnAdapterHF import AttnAdapterHF
from pope_evaluator import evaluate_pope, extract_prediction


def inject_causallens_adapters(
    model: LlavaForConditionalGeneration,
    layer_start: int = 10,
    layer_end: int = 20,
    lambda_causal: float = 0.15,
    gamma_mix: float = 0.15,
    sys_len: int = 35,
    img_len: int = 576,
) -> List[AttnAdapterHF]:
    """
    Inject AttnAdapterHF into language model decoder layers [layer_start, layer_end] (inclusive).
    """
    adapters = []
    # In LlavaForConditionalGeneration, language model is typically at model.language_model.model.layers or model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers = model.model.language_model.layers
    else:
        raise AttributeError("Could not find transformer layers in Llava model.")

    for i, layer in enumerate(layers):
        if layer_start <= i <= layer_end:
            print(f"[CausalLens LLaVA] Replacing layer {i} self_attn with AttnAdapterHF")
            orig_attn = layer.self_attn
            config = getattr(orig_attn, "config", model.config)
            
            adapter = AttnAdapterHF(
                config=config,
                layer_idx=i,
                lambda_causal=lambda_causal,
                gamma_mix=gamma_mix,
                sys_len=sys_len,
                img_len=img_len,
            )
            # Copy projection weights
            adapter.q_proj.load_state_dict(orig_attn.q_proj.state_dict())
            adapter.k_proj.load_state_dict(orig_attn.k_proj.state_dict())
            adapter.v_proj.load_state_dict(orig_attn.v_proj.state_dict())
            adapter.o_proj.load_state_dict(orig_attn.o_proj.state_dict())

            if hasattr(orig_attn, "rotary_emb"):
                adapter.rotary_emb = orig_attn.rotary_emb

            # Preserve accelerate multi-GPU hooks if present
            if hasattr(orig_attn, "_hf_hook"):
                adapter._hf_hook = orig_attn._hf_hook

            # Move to target layer device and dtype
            param = next(orig_attn.parameters())
            adapter = adapter.to(dtype=param.dtype, device=param.device)
            
            layer.self_attn = adapter
            adapters.append(adapter)

    print(f"[CausalLens LLaVA] Successfully injected {len(adapters)} AttnAdapterHF modules.")
    return adapters


def run_llava_pope(
    model_path: str = "llava-hf/llava-1.5-7b-hf",
    pope_dir: str = "experiments/data/POPE/coco",
    image_dir: Optional[str] = None,
    output_dir: str = "results",
    split: str = "all",
    lambda_causal: float = 0.15,
    gamma_mix: float = 0.15,
    layer_start: int = 10,
    layer_end: int = 20,
    sys_len: int = 35,
    img_len: int = 576,
    max_new_tokens: int = 6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run POPE evaluation with CausalLens intervention on LLaVA-1.5-7B.
    """
    set_seed(seed)
    
    # 1. Resolve image and annotation paths
    image_dir = find_coco_val2014_dir(image_dir)
    print(f"[LLaVA Runner] COCO val2014 directory: {image_dir}")
    
    splits = ["random", "popular", "adversarial"] if split == "all" else [split]
    for s in splits:
        pope_file = os.path.join(pope_dir, f"coco_pope_{s}.json")
        if not os.path.exists(pope_file):
            raise FileNotFoundError(f"POPE annotation file not found: {pope_file}")

    # 2. Precision & Model Loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[LLaVA Runner] Loading {model_path} with dtype={torch_dtype}, device_map='auto'...")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # 3. Inject CausalLens AttnAdapterHF
    adapters = inject_causallens_adapters(
        model=model,
        layer_start=layer_start,
        layer_end=layer_end,
        lambda_causal=lambda_causal,
        gamma_mix=gamma_mix,
        sys_len=sys_len,
        img_len=img_len,
    )

    all_results = {}
    image_token_id = getattr(model.config, "image_token_index", 32000)

    # 4. Iterate through splits
    for s in splits:
        pope_file = os.path.join(pope_dir, f"coco_pope_{s}.json")
        split_out_dir = os.path.join(output_dir, s)
        os.makedirs(split_out_dir, exist_ok=True)
        raw_output_path = os.path.join(split_out_dir, "raw_outputs.jsonl")
        metrics_output_path = os.path.join(split_out_dir, "metrics.json")
        config_output_path = os.path.join(split_out_dir, "run_config.json")

        # Save run config
        run_config = {
            "model": "llava",
            "model_path": model_path,
            "split": s,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "lambda_causal": lambda_causal,
            "gamma_mix": gamma_mix,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "sys_len": sys_len,
            "img_len": img_len,
            "seed": seed,
            "torch_dtype": str(torch_dtype),
            "attn_implementation": "eager",
        }
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        # Load split questions
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

        print(f"\n[LLaVA Runner] Processing split: '{s}' ({len(samples)} samples, {len(finished_qids)} already done) -> {raw_output_path}")

        cached_image = None
        cached_image_path = None

        with open(raw_output_path, "a", encoding="utf-8") as f_out:
            for item in tqdm(samples, desc=f"LLaVA POPE ({s})"):
                qid = item["question_id"]
                if qid in finished_qids:
                    continue
                image_name = item["image"]
                question = item["text"]
                label = item["label"]

                image_path = os.path.join(image_dir, image_name)
                if not os.path.exists(image_path):
                    continue

                if image_path == cached_image_path and cached_image is not None:
                    image = cached_image
                else:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        cached_image = image
                        cached_image_path = image_path
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        continue
                # Construct prompt with conversation format to maintain sys_len token prefix
                prompt = (
                    "A chat between a curious human and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                    f"USER: <image>\n{question} ASSISTANT:"
                )

                inputs = processor(text=prompt, images=image, return_tensors="pt")
                # Move to proper device
                for k in inputs:
                    if hasattr(inputs[k], "to"):
                        if k == "pixel_values":
                            inputs[k] = inputs[k].to(dtype=model.dtype, device=model.device)
                        else:
                            inputs[k] = inputs[k].to(device=model.device)

                # Update token range for adapters
                input_ids = inputs["input_ids"]
                if (input_ids[0] == image_token_id).any():
                    vis_start = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0].item()
                else:
                    vis_start = sys_len
                for adapter in adapters:
                    adapter.update_token_range(sys_len=vis_start, img_len=img_len)

                # Generate with strict Greedy Decoding (max_new_tokens=6)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                    )

                in_len = inputs["input_ids"].shape[-1]
                generated_tokens = output_ids[0, in_len:]
                output_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()
                pred_decision = extract_prediction(output_text)

                result_entry = {
                    "question_id": qid,
                    "image": image_name,
                    "question": question,
                    "label": label,
                    "pred": pred_decision,
                    "answer": output_text,
                    "text": output_text,
                }
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                f_out.flush()

                # Clean cache periodically to avoid VRAM fragmentation across 3000 questions
                if torch.cuda.is_available() and (item.get("question_id", 0) % 500 == 0):
                    torch.cuda.empty_cache()

        # Evaluate split
        metrics = evaluate_pope(pope_file, raw_output_path, metrics_output_path)
        all_results[s] = {
            "metrics": metrics,
            "raw_output": raw_output_path,
            "config": run_config,
        }
        print(f"[LLaVA Runner] Split '{s}' Results:")
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall   : {metrics['recall']*100:.2f}%")
        print(f"  F1-Score : {metrics['f1']*100:.2f}%")
        print(f"  Yes-Ratio: {metrics['yes_ratio']*100:.2f}%")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-1.5 CausalLens POPE Evaluation")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--pope_dir", type=str, default="experiments/data/POPE/coco")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/llava_pope")
    parser.add_argument("--split", type=str, default="all", choices=["random", "popular", "adversarial", "all"])
    parser.add_argument("--lambda_causal", type=float, default=0.15)
    parser.add_argument("--gamma_mix", type=float, default=0.15)
    parser.add_argument("--layer_start", type=int, default=10)
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument("--sys_len", type=int, default=35)
    parser.add_argument("--img_len", type=int, default=576)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_llava_pope(
        model_path=args.model_path,
        pope_dir=args.pope_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        split=args.split,
        lambda_causal=args.lambda_causal,
        gamma_mix=args.gamma_mix,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        sys_len=args.sys_len,
        img_len=args.img_len,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
