import os
import glob
from typing import Optional


def find_coco_val2014_dir(preferred_path: Optional[str] = None) -> str:
    """
    Auto-detect the COCO val2014 images directory.
    
    Priority:
    1. preferred_path (if provided and directory exists with COCO images)
    2. Common Kaggle dataset paths:
       - /kaggle/input/datasets/biminhco/val2014/val2014
       - /kaggle/input/val2014/val2014
       - /kaggle/input/coco-2014-val/val2014
       - /kaggle/input/coco2014-val/val2014
       - /kaggle/input/mscoco2014/val2014
    3. Recursive scan under /kaggle/input (max depth 4) looking for COCO_val2014_*.jpg
    4. Local fallback: experiments/data/COCO/val2014
    """
    # 1. Check user-supplied path
    if preferred_path and os.path.isdir(preferred_path):
        # Verify it has jpgs or at least looks like val2014
        sample_files = glob.glob(os.path.join(preferred_path, "COCO_val2014_*.jpg"))
        if sample_files or len(os.listdir(preferred_path)) > 0:
            print(f"[COCO Path Finder] Using specified path: {preferred_path}")
            return os.path.abspath(preferred_path)

    # 2. Known standard Kaggle paths
    known_kaggle_paths = [
        "/kaggle/input/datasets/biminhco/val2014/val2014",
        "/kaggle/input/val2014/val2014",
        "/kaggle/input/val2014",
        "/kaggle/input/coco-2014-val/val2014",
        "/kaggle/input/coco2014-val/val2014",
        "/kaggle/input/mscoco2014/val2014",
        "/kaggle/input/coco-val2014/val2014",
    ]
    for path in known_kaggle_paths:
        if os.path.isdir(path):
            sample_files = glob.glob(os.path.join(path, "COCO_val2014_*.jpg"))
            if sample_files:
                print(f"[COCO Path Finder] Found COCO val2014 at known path: {path}")
                return path

    # 3. Dynamic search under /kaggle/input
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            # Check if any COCO_val2014 file exists in this directory
            for f in files:
                if f.startswith("COCO_val2014_") and f.endswith(".jpg"):
                    print(f"[COCO Path Finder] Discovered COCO val2014 directory: {root}")
                    return root
            # Limit search depth to avoid slow traversal
            depth = root[len(kaggle_input):].count(os.sep)
            if depth >= 4:
                dirs.clear()

    # 4. Local repo fallback
    local_candidates = [
        os.path.join(os.getcwd(), "experiments", "data", "COCO", "val2014"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "data", "COCO", "val2014"),
    ]
    for cand in local_candidates:
        if os.path.isdir(cand):
            print(f"[COCO Path Finder] Found local COCO directory: {cand}")
            return os.path.abspath(cand)

    raise FileNotFoundError(
        "Could not locate COCO val2014 images directory! "
        "Please provide --image_dir <path_to_val2014> or attach the COCO val2014 dataset to your Kaggle Notebook."
    )
