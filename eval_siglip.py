#!/usr/bin/env python3
#
# API Key: export GEMINI_API_KEY="AIzaSyAfGTQnXSxX2yeGaNGZ-PN9V549QWPA7ik"
#
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple, Optional
from tqdm import tqdm

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Each line must be a JSON object.")
                yield obj
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Dataset parsing
# -----------------------------
def extract_sets(sample: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Expect sample format:
      {
        "id": ...,
        "image_path": ...,
        "sets": [
          {"tag": "...", "gt": "...", "neg": ["...","...","...","..."]},
          ...
        ]
      }
    """
    sid = str(sample.get("id", ""))
    img_path = sample.get("image_path", "")
    sets = sample.get("sets", [])
    if not isinstance(img_path, str) or not img_path:
        raise ValueError("missing image_path")
    if not isinstance(sets, list) or len(sets) == 0:
        raise ValueError("missing sets")
    return sid, img_path, sets


def validate_set_item(it: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    tag = it.get("tag", "")
    gt = it.get("gt", "")
    neg = it.get("neg", [])
    if not isinstance(tag, str) or not tag:
        raise ValueError("bad tag")
    if not isinstance(gt, str) or not gt.strip():
        raise ValueError("bad gt")
    if not isinstance(neg, list) or len(neg) != 4 or any((not isinstance(x, str) or not x.strip()) for x in neg):
        raise ValueError("bad neg")
    return tag, gt.strip(), [x.strip() for x in neg]


# -----------------------------
# Eval core
# -----------------------------
@torch.inference_mode()
def score_one_set(
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
    image: Image.Image,
    texts: List[str],
) -> torch.Tensor:
    """
    Returns logits_per_image: shape (1, len(texts))
    """
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.logits_per_image  # (1, T)


def open_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate SigLIP-style models on your 5-way sets (GT + 4 Neg) accuracy."
    )
    ap.add_argument("--data", required=True, help="JSONL dataset path (each line has image_path and sets).")
    ap.add_argument("--model", required=True, help='HF model repo/path for AutoModel.from_pretrained(...)')
    ap.add_argument("--processor", required=True, help='HF processor repo/path for AutoProcessor.from_pretrained(...)')
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Use cuda if available.")
    ap.add_argument("--batch-text", action="store_true",
                    help="(Optional) Process multiple sets per image in one forward by concatenating texts. "
                         "Saves time but slightly more code; default is simple per-set forward.")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no limit; otherwise eval first N samples.")
    ap.add_argument("--save-errors", default="", help="Optional output JSONL to save wrong predictions.")
    ap.add_argument("--topk", type=int, default=5, help="When saving errors, save topk ranked texts.")
    args = ap.parse_args()

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # load model/processor (SigLIP model)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)

    model.to(device)
    model.eval()

    # stats
    total_sets = 0
    correct_sets = 0
    per_tag_total: Dict[str, int] = {}
    per_tag_correct: Dict[str, int] = {}

    error_rows: List[Dict[str, Any]] = []

    pbar = tqdm(read_jsonl(args.data), desc="samples")
    for idx, sample in enumerate(pbar, 1):
        if args.max_samples and idx > args.max_samples:
            break

        # allow skipping lines that are error records
        if "error" in sample and "sets" not in sample:
            continue

        try:
            sid, img_path, sets = extract_sets(sample)
        except Exception:
            continue

        if not os.path.exists(img_path):
            continue

        try:
            image = open_image_rgb(img_path)
        except Exception:
            continue

        # simple path: per set forward
        if not args.batch_text:
            for it in sets:
                try:
                    tag, gt, negs = validate_set_item(it)
                except Exception:
                    continue

                texts = [gt] + negs  # GT index = 0
                logits = score_one_set(model, processor, device, image, texts)  # (1, 5)
                pred = int(torch.argmax(logits, dim=1).item())
                is_correct = (pred == 0)

                total_sets += 1
                correct_sets += int(is_correct)
                per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
                per_tag_correct[tag] = per_tag_correct.get(tag, 0) + int(is_correct)

                if (not is_correct) and args.save_errors:
                    scores = logits.squeeze(0).detach().float().cpu()
                    # rank
                    topk = min(args.topk, len(texts))
                    order = torch.argsort(scores, descending=True)[:topk].tolist()
                    error_rows.append({
                        "id": sid,
                        "image_path": img_path,
                        "tag": tag,
                        "gt": gt,
                        "neg": negs,
                        "pred_index": pred,
                        "pred_text": texts[pred],
                        "scores": [float(scores[i]) for i in range(len(texts))],
                        "topk": [{"idx": i, "text": texts[i], "score": float(scores[i])} for i in order],
                    })

        else:
            # faster mode: concat all texts for this image, then slice back per set
            # (still exact same metric)
            all_texts: List[str] = []
            meta: List[Tuple[str, str, List[str], int]] = []  # (tag, gt, negs, start_idx)
            for it in sets:
                try:
                    tag, gt, negs = validate_set_item(it)
                except Exception:
                    continue
                start = len(all_texts)
                all_texts.extend([gt] + negs)
                meta.append((tag, gt, negs, start))

            if not meta:
                continue

            logits_all = score_one_set(model, processor, device, image, all_texts).squeeze(0).detach().float().cpu()
            for (tag, gt, negs, start) in meta:
                scores = logits_all[start:start+5]
                pred = int(torch.argmax(scores).item())
                is_correct = (pred == 0)

                total_sets += 1
                correct_sets += int(is_correct)
                per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
                per_tag_correct[tag] = per_tag_correct.get(tag, 0) + int(is_correct)

                if (not is_correct) and args.save_errors:
                    texts = [gt] + negs
                    topk = min(args.topk, 5)
                    order = torch.argsort(scores, descending=True)[:topk].tolist()
                    error_rows.append({
                        "id": sid,
                        "image_path": img_path,
                        "tag": tag,
                        "gt": gt,
                        "neg": negs,
                        "pred_index": pred,
                        "pred_text": texts[pred],
                        "scores": [float(scores[i]) for i in range(5)],
                        "topk": [{"idx": i, "text": texts[i], "score": float(scores[i])} for i in order],
                    })

        # update progress bar
        acc = (correct_sets / total_sets) if total_sets else 0.0
        pbar.set_postfix({"set_acc": f"{acc:.4f}", "sets": total_sets})

    # print report
    overall_acc = (correct_sets / total_sets) if total_sets else 0.0
    print("\n=== Results ===")
    print(f"Total sets: {total_sets}")
    print(f"Correct sets: {correct_sets}")
    print(f"Set-level Acc (GT ranked #1 among 5): {overall_acc:.6f}")

    print("\n--- Per-tag Acc ---")
    for tag in sorted(per_tag_total.keys()):
        t = per_tag_total[tag]
        c = per_tag_correct.get(tag, 0)
        print(f"{tag:>10s}: {c}/{t} = {c/t:.6f}")

    # save errors
    if args.save_errors:
        write_jsonl(args.save_errors, error_rows)
        print(f"\nSaved wrong cases to: {args.save_errors}  (rows={len(error_rows)})")


if __name__ == "__main__":
    main()
