# CausalLens POPE Benchmark on Kaggle (2× NVIDIA T4 GPUs)

Hướng dẫn chạy benchmark **POPE (Random, Popular, Adversarial)** cho **CausalLens** trên môi trường **Kaggle Notebooks (2× GPU T4 miễn phí)**.

---

## 1. Thiết lập Môi trường Kaggle

1. **Tạo Kaggle Notebook mới:**
   - Accelerator: Chọn **GPU T4 x2**.
   - Internet: Bật **Always ON**.
   - Environment: **Python 3.10 / Latest**.

2. **Gắn Dataset COCO val2014:**
   - Tìm kiếm dataset `COCO 2014 val` hoặc `val2014` (ví dụ: `biminhco/val2014`).
   - Gắn vào notebook (path thường sẽ là `/kaggle/input/.../val2014`).
   - Module `causallens_utils/coco_path_finder.py` sẽ tự động quét và phát hiện đường dẫn ảnh.

3. **Cấu hình HuggingFace Secrets:**
   - Vào menu `Add-ons` -> `Secrets`.
   - Thêm secret tên `HF_TOKEN` chứa HuggingFace Access Token của bạn.

---

## 2. Các Ràng buộc Benchmark POPE Bắt buộc

Tuân thủ nghiêm ngặt các điều kiện chuẩn hóa từ `HANDOVER.md`:
- **Max New Tokens:** `max_new_tokens = 6`.
- **Decoding Strategy:** **Greedy Decoding** (`do_sample=False`, `temperature=0.0`).
- **Precision:** `torch.bfloat16` nếu GPU hỗ trợ (fallback `torch.float16`).
- **Multi-GPU:** `device_map="auto"` phân bổ đều trên 2× GPU T4 (32GB tổng VRAM).
- **Prompt Suffix cho Qwen2-VL:** Bắt buộc có hậu tố: `"Please answer with yes or no."`
- **Attention Implementation:** `attn_implementation="eager"` để hook CausalLens can thiệp attention trực tiếp.
- **Tối ưu tốc độ Qwen2-VL:** Giới hạn `max_pixels: 313600` trong processor.

---

## 3. Cách Sử dụng

### Cách 1: Sử dụng Notebook `kaggle_causallens_pope.ipynb` (Khuyên dùng)
Upload trực tiếp file [`kaggle_causallens_pope.ipynb`](kaggle_causallens_pope.ipynb) lên Kaggle và chạy tuần tự 6 cells:
- **Cell 1:** Cài đặt dependencies và kiểm tra GPU / CUDA / BF16.
- **Cell 2:** Đăng nhập HuggingFace qua Secrets.
- **Cell 3:** Clone repo & auto-detect COCO val2014 path.
- **Cell 4:** Chọn mô hình (`MODEL = "qwen2vl"` hoặc `MODEL = "llava"`).
- **Cell 5:** Chạy benchmark tự động cho cả 3 splits (`random`, `popular`, `adversarial`).
- **Cell 6:** Hiển thị bảng tổng kết kết quả (Accuracy, Precision, Recall, F1, Yes-Ratio, Unknowns).

---

### Cách 2: Chạy qua Unified CLI `kaggle_pope_runner.py`

#### Chạy Qwen2-VL-7B-Instruct (Full 3 splits):
```bash
python kaggle_pope_runner.py \
    --model qwen2vl \
    --split all \
    --lambda_causal 0.15 \
    --gamma_mix 0.15 \
    --layer_start 10 \
    --layer_end 20 \
    --max_new_tokens 6 \
    --seed 42
```

#### Chạy LLaVA-1.5-7B (Full 3 splits):
```bash
python kaggle_pope_runner.py \
    --model llava \
    --split all \
    --lambda_causal 0.15 \
    --gamma_mix 0.15 \
    --layer_start 10 \
    --layer_end 20 \
    --max_new_tokens 6 \
    --seed 42
```

#### Đánh giá file kết quả độc lập qua `pope_evaluator.py`:
```bash
python pope_evaluator.py \
    --gt_path experiments/data/POPE/coco/coco_pope_random.json \
    --pred_path results/qwen2vl_pope_xxx/random/raw_outputs.jsonl \
    --output_path results/qwen2vl_pope_xxx/random/metrics.json
```

---

## 4. Cấu trúc Thư mục Kết quả

```
results/
└── {model}_pope_{timestamp}/
    ├── random/
    │   ├── raw_outputs.jsonl   # Dự đoán từng câu hỏi (3000 mẫu)
    │   ├── metrics.json        # Acc, Prec, Recall, F1, Yes-Ratio
    │   └── run_config.json     # Hyperparameters và model config
    ├── popular/
    │   ├── raw_outputs.jsonl
    │   ├── metrics.json
    │   └── run_config.json
    ├── adversarial/
    │   ├── raw_outputs.jsonl
    │   ├── metrics.json
    │   └── run_config.json
    └── summary.json            # Tổng hợp cả 3 splits
```

---

## 5. Các Lỗi Môi trường & Dependency Tiềm ẩn trên Kaggle và Cách Khắc phục

### 1. Tesla T4 không hỗ trợ BF16 phần cứng
- **Hiện tượng:** Nếu ép dùng `torch.bfloat16`, PyTorch trên GPU Turing (sm_75) sẽ phải giả lập phần mềm, gây chậm từ 3× đến 5× hoặc báo lỗi `RuntimeError: "addmm_cuda" not implemented for 'BFloat16'`.
- **Giải pháp trong code:** Hệ thống tự động kiểm tra `torch.cuda.is_bf16_supported()`. Trên Tesla T4 (trả về `False`), code tự động fallback an toàn sang `torch.float16` có tăng tốc phần cứng Tensor Cores. Đồng thời, cả `AttnAdapterHF.py` và `qwen2ours_pope.py` đều có lớp lọc `isinf` ngăn tràn số trước Softmax trong FP16.

### 2. Xung đột CUDA Driver do `torchaudio`
- **Hiện tượng:** Kaggle image gốc đôi khi có `torchaudio` cài với CUDA version khác, gây warning hoặc lỗi khởi tạo CUDA.
- **Giải pháp:** Cell 1 chủ động gỡ bỏ `!pip uninstall -y -q torchaudio` trước khi cài các package khác và tuyệt đối không `pip install torch torchvision` để giữ nguyên CUDA wheels chuẩn của Kaggle.

### 3. Không cài đặt được `flash-attn`
- **Hiện tượng:** Lỗi build wheel thất bại hoặc runtime error: `FlashAttention only supports Ampere, Ada, or Hopper GPUs`.
- **Giải pháp:** Tesla T4 (Turing compute capability 7.5) không hỗ trợ FlashAttention. Toàn bộ codebase đã cấu hình `attn_implementation="eager"`, vừa tương thích tuyệt đối với T4 vừa cho phép CausalLens can thiệp attention trực tiếp.

### 4. Mất hook của `accelerate` khi can thiệp Attention trên Multi-GPU
- **Hiện tượng:** `RuntimeError: Expected all tensors to be on the same device` khi mô hình sharded trên 2 GPU (`cuda:0` và `cuda:1`).
- **Giải pháp:** Khi thay thế `layer.self_attn` bằng `AttnAdapter`, code chủ động bảo toàn `attn_adapter._hf_hook = layer.self_attn._hf_hook` để cơ chế multi-GPU của `accelerate` tiếp tục hoạt động chính xác.

### 5. Tràn VRAM (CUDA OOM) do phân mảnh bộ nhớ qua 9,000 câu hỏi
- **Hiện tượng:** `torch.cuda.OutOfMemoryError` sau khi chạy liên tục qua nhiều split.
- **Giải pháp:**
  - Giới hạn `max_pixels: 313600` cho Qwen2-VL để kiểm soát số lượng visual tokens.
  - Sử dụng `torch.inference_mode()` loại bỏ lưu graph đạo hàm.
  - Tự động gọi `torch.cuda.empty_cache()` định kỳ mỗi 500 câu hỏi.

### 6. Tràn ổ đĩa `/kaggle/working` (Quota 20GB)
- **Hiện tượng:** `OSError: [Errno 28] No space left on device`.
- **Giải pháp:** Giữ cache HuggingFace tại thư mục gốc `/root/.cache/huggingface` (có ~73GB trống), không chuyển vào `/kaggle/working`. Chỉ chạy 1 mô hình mỗi session (hoặc xóa checkpoint cũ nếu chạy cả 2).

### 7. Mất kết nối tab trình duyệt (Session Timeout)
- **Hiện tượng:** Session interactive của Kaggle tự động dừng nếu mất kết nối hoặc đóng tab quá 60 phút.
- **Giải pháp:** Khuyên dùng chế độ **"Save & Run All (Commit)"** trên Kaggle. Quá trình benchmark sẽ chạy ngầm tối đa 12 tiếng trên máy chủ độc lập với trình duyệt.
