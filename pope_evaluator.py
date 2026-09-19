import os
import json
import argparse
from typing import Dict, Any, Optional, List


def parse_pope_file(file_path: str) -> List[Dict[str, Any]]:
    """Load POPE entries from JSONL or JSON array format."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("[") and content.endswith("]"):
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                pass
        # Fallback to json lines
        for line in content.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_prediction(raw_text: str) -> str:
    """Extract yes/no prediction from raw generated text."""
    text_lower = raw_text.lower().strip()
    
    # Check simple presence (consistent with POPE standard protocol)
    # Check both tokens carefully
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    
    if has_yes and not has_no:
        return "yes"
    elif has_no and not has_yes:
        return "no"
    elif has_yes and has_no:
        # If both present, take whichever appears first
        yes_idx = text_lower.find("yes")
        no_idx = text_lower.find("no")
        return "yes" if yes_idx < no_idx else "no"
    else:
        return "unknown"


def evaluate_pope(gt_path: str, pred_path: str, output_metrics_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate POPE predictions against ground truth.
    
    Args:
        gt_path: Path to ground truth JSON/JSONL file (contains label, question_id)
        pred_path: Path to prediction JSONL file (contains text/answer/pred, question_id)
        output_metrics_path: Optional path to save metrics JSON
        
    Returns:
        dict containing: accuracy, precision, recall, f1, yes_ratio, unknowns, total, etc.
    """
    gt_data = parse_pope_file(gt_path)
    pred_data = parse_pope_file(pred_path)
    
    gt_by_id = {item["question_id"]: item for item in gt_data if "question_id" in item}
    
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknowns = 0
    yes_answers = 0
    total = len(pred_data)
    
    for i, pred_item in enumerate(pred_data):
        qid = pred_item.get("question_id")
        if qid in gt_by_id:
            gt_item = gt_by_id[qid]
        elif i < len(gt_data):
            gt_item = gt_data[i]
        else:
            continue
            
        gt_label = str(gt_item.get("label", "")).lower().strip()
        
        # Get raw response string from pred_item
        raw_pred = ""
        for key in ["pred", "text", "answer", "response"]:
            if key in pred_item and pred_item[key] is not None:
                raw_pred = str(pred_item[key])
                break
                
        pred_decision = extract_prediction(raw_pred)
        
        if pred_decision == "yes":
            yes_answers += 1
            if gt_label == "yes":
                true_pos += 1
            else:
                false_pos += 1
        elif pred_decision == "no":
            if gt_label == "no":
                true_neg += 1
            else:
                false_neg += 1
        else:
            unknowns += 1
            # Unknown is counted as a failure to answer correctly
            if gt_label == "yes":
                false_neg += 1
            else:
                false_pos += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_pos + true_neg) / total if total > 0 else 0.0
    yes_ratio = yes_answers / total if total > 0 else 0.0
    unknown_ratio = unknowns / total if total > 0 else 0.0
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_ratio, 4),
        "unknown_ratio": round(unknown_ratio, 4),
        "unknowns": unknowns,
        "total": total,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
    }
    
    if output_metrics_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_metrics_path)), exist_ok=True)
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate POPE benchmark predictions")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth JSON/JSONL file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to prediction JSONL file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save output metrics JSON")
    args = parser.parse_args()
    
    results = evaluate_pope(args.gt_path, args.pred_path, args.output_path)
    print("=" * 50)
    print("POPE Evaluation Metrics:")
    for k, v in results.items():
        if isinstance(v, float) and "ratio" in k or k in ["accuracy", "precision", "recall", "f1"]:
            print(f"  {k:15s}: {v*100:.2f}%")
        else:
            print(f"  {k:15s}: {v}")
    print("=" * 50)
