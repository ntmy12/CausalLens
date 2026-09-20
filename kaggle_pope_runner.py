import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_summary_table(model_name: str, split_results: Dict[str, Any]):
    """Print an ASCII summary table of POPE benchmark metrics."""
    header = (
        f"\n{'='*76}\n"
        f"🏆 CausalLens POPE Benchmark Summary - {model_name.upper()}\n"
        f"{'='*76}\n"
        f"| {'Split':<14} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Yes-Ratio':<10} |\n"
        f"|{'-'*16}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|"
    )
    print(header)
    for split_name, data in split_results.items():
        m = data.get("metrics", {})
        acc = f"{m.get('accuracy', 0.0)*100:.2f}%"
        prec = f"{m.get('precision', 0.0)*100:.2f}%"
        rec = f"{m.get('recall', 0.0)*100:.2f}%"
        f1 = f"{m.get('f1', 0.0)*100:.2f}%"
        yes = f"{m.get('yes_ratio', 0.0)*100:.2f}%"
        print(f"| {split_name.capitalize():<14} | {acc:<10} | {prec:<10} | {rec:<10} | {f1:<10} | {yes:<10} |")
    print(f"{'='*76}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified CausalLens POPE Benchmark Runner for Kaggle (2x GPU T4)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llava", "qwen2vl"],
        help="Target VLM to evaluate: 'llava' (LLaVA-1.5-7B) or 'qwen2vl' (Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["random", "popular", "adversarial", "all"],
        help="POPE split to evaluate (default: 'all')",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="HF hub model ID or local directory. Default auto-selected by model type.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to COCO val2014 images directory. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--pope_dir",
        type=str,
        default="experiments/data/POPE/coco",
        help="Directory containing coco_pope_*.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Root directory for saving outputs and metrics.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Explicit output directory (supports resuming an existing interrupted run).",
    )
    parser.add_argument(
        "--lambda_causal",
        type=float,
        default=0.15,
        help="Causal intervention strength (default: 0.15)",
    )
    parser.add_argument(
        "--gamma_mix",
        type=float,
        default=0.15,
        help="Mixing ratio between original attention and causal intervention (default: 0.15)",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=10,
        help="Starting layer for CausalLens adapter injection (default: 10)",
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=20,
        help="Ending layer for CausalLens adapter injection (default: 20)",
    )
    parser.add_argument(
        "--sys_len",
        type=int,
        default=None,
        help="System token count (defaults: 35 for LLaVA, 31 for Qwen2-VL)",
    )
    parser.add_argument(
        "--img_len",
        type=int,
        default=None,
        help="Image token count (defaults: 576 for LLaVA, 256 for Qwen2-VL)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=6,
        help="Maximum new tokens generated (POPE benchmark condition: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Create or use run folder (supports resuming)
    if args.run_dir:
        run_output_dir = os.path.abspath(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(args.output_dir, f"{args.model}_pope_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print(f"🚀 Launching CausalLens POPE Benchmark | Model: {args.model.upper()}")
    print(f"📁 Output Directory: {run_output_dir}")
    print(f"⚡ Max New Tokens: {args.max_new_tokens} (Greedy Decoding)")
    print(f"🔧 Hyperparameters: lambda={args.lambda_causal}, gamma={args.gamma_mix}, layers=[{args.layer_start}, {args.layer_end}]")
    print("=" * 76 + "\n")

    if args.model == "llava":
        from llava_pope_runner import run_llava_pope
        model_path = args.model_path or "llava-hf/llava-1.5-7b-hf"
        sys_len = args.sys_len if args.sys_len is not None else 35
        img_len = args.img_len if args.img_len is not None else 576

        results = run_llava_pope(
            model_path=model_path,
            pope_dir=args.pope_dir,
            image_dir=args.image_dir,
            output_dir=run_output_dir,
            split=args.split,
            lambda_causal=args.lambda_causal,
            gamma_mix=args.gamma_mix,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            sys_len=sys_len,
            img_len=img_len,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    elif args.model == "qwen2vl":
        from qwen2ours_pope import run_qwen2vl_pope
        model_path = args.model_path or "Qwen/Qwen2-VL-7B-Instruct"

        results = run_qwen2vl_pope(
            model_path=model_path,
            pope_dir=args.pope_dir,
            image_dir=args.image_dir,
            output_dir=run_output_dir,
            split=args.split,
            lambda_causal=args.lambda_causal,
            gamma_mix=args.gamma_mix,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Save consolidated summary
    summary_path = os.path.join(run_output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print_summary_table(args.model, results)
    print(f"✅ POPE Benchmark Complete! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
