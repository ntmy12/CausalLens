# HANDOVER & CONTEXT
## 1. Goal
Evaluate hallucination mitigation methods (CausalLens, ONLY, OPERA) on VLMs (Qwen2-VL, LLaVA) using BEAF/POPE benchmarks.

## 2. Environment (CRITICAL)
- **Path:** `/home/nvidia-lab/ai4life/phuongnh/vlm-truth`
- **Conda Env:** `vlm_truth_py313` -> absolute path: `/home/nvidia-lab/miniconda3/envs/vlm_truth_py313/bin/python`
- **GPU:** H100 80GB (CUDA_VISIBLE_DEVICES=0)
- **HF_HOME:** `export HF_HOME="/home/nvidia-lab/data_mount/huggingface_cache"` (Mandatory to avoid full disk)

## 3. CausalLens Modifications Applied
- Forced `config._attn_implementation = "eager"` in `qwen2vl_wrapper.py` to allow hook intervention.
- Restricted `max_pixels: 313600` to speed up generation (down from 55 hours to ~10 hours).
- Confirmed LLaVA natively runs Eager properly with `attn_impl="eager"`.

## 4. POPE Evaluation Conditions & Configuration (BẮT BUỘC)
- **Max New Tokens:** `max_new_tokens = 6`.
- **Decoding Strategy:** **Greedy Decoding** (`do_sample=False`, `temperature=0.0`).
- **Prompt Suffix for QwenVL:** Nếu chạy mô hình QwenVL (Qwen2-VL), bắt buộc thêm hậu tố (suffix) vào sau câu hỏi: `"Please answer with yes or no."`
  - Ví dụ: `prompt = f"{question} Please answer with yes or no."`

## 5. Next Tasks for Agent
1. Clone OPERA repo: `git clone https://github.com/ntmy12/OPERA.git`
2. Integrate OPERA decoding with Qwen2-VL in the existing framework.
3. Run the evaluation on **POPE** (splits: `random`, `popular`, `adversarial`) tuân thủ nghiêm ngặt các điều kiện: `max_new_tokens=6`, Greedy decoding, và prompt suffix `"Please answer with yes or no."` cho Qwen2-VL.
