#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import Dict, Any, Iterable, Optional, List, Tuple


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


def write_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def index_images(images_root: str) -> Dict[str, str]:
    """
    Build mapping: image filename -> absolute path
    Example: "qual_dev_000123.jpg" -> "/.../docci_images/qual_dev_000123.jpg"
    """
    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(images_root):
        for fn in files:
            low = fn.lower()
            if low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png"):
                mapping.setdefault(fn, os.path.join(root, fn))
    return mapping


def build_description_index(
    descriptions_path: str,
    user_keep_invalid: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Index descriptions by example_id.
    NOTE: DOCCI normally has unique example_id; if duplicates exist, last wins.
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for d in read_jsonl(descriptions_path):
        ex_id = d.get("example_id")
        image_file = d.get("image_file")
        text = d.get("description")
        split = d.get("split")

        if isinstance(ex_id, str) and isinstance(image_file, str) and isinstance(text, str):
            idx[ex_id] = {
                "example_id": ex_id,
                "image_file": image_file,
                "description": text,
                "split": split,
            }
        elif user_keep_invalid and isinstance(ex_id, str):
            # 保留但标记 invalid（一般不需要）
            idx[ex_id] = {
                "example_id": ex_id,
                "image_file": image_file,
                "description": text,
                "split": split,
                "_invalid_desc": True,
            }
    return idx


def build_metadata_index(
    metadata_path: Optional[str],
    keep_cloud_vision: bool,
    keep_dsg: bool,
) -> Dict[str, Dict[str, Any]]:
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    if not metadata_path:
        return meta_by_id

    for m in read_jsonl(metadata_path):
        ex_id = m.get("example_id")
        if not isinstance(ex_id, str):
            continue

        # 可选：丢掉超大字段
        if not keep_cloud_vision:
            m.pop("cloud_vision_api_responses", None)
        if not keep_dsg:
            m.pop("dsg", None)

        meta_by_id[ex_id] = m
    return meta_by_id


def choose_ids(
    all_ids: List[str],
    id_to_split: Dict[str, str],
    limit: Optional[int],
    seed: int,
    stratify_by_split: bool,
) -> List[str]:
    """
    Choose subset of ids.
    - If limit is None: return all_ids (original order)
    - Else: sample limit ids (random), optionally stratified by split.
    """
    if limit is None:
        return all_ids

    if limit <= 0:
        return []

    if len(all_ids) < limit:
        raise RuntimeError(f"Not enough valid samples to sample: have {len(all_ids)}, need {limit}")

    rnd = random.Random(seed)

    if not stratify_by_split:
        return rnd.sample(all_ids, limit)

    # stratified proportional sampling
    by_split: Dict[str, List[str]] = {}
    for ex_id in all_ids:
        s = id_to_split.get(ex_id, "unknown")
        by_split.setdefault(s, []).append(ex_id)

    total = len(all_ids)
    # initial allocation (proportional, ensure at least 1 if split exists)
    alloc: Dict[str, int] = {}
    for s, ids in by_split.items():
        k_s = round(len(ids) / total * limit)
        alloc[s] = max(1, k_s)

    # fix sum to exactly limit
    def cur_sum() -> int:
        return sum(alloc.values())

    # if too many, reduce from largest alloc buckets
    while cur_sum() > limit:
        s = max(alloc, key=lambda x: alloc[x])
        if alloc[s] > 1:
            alloc[s] -= 1
        else:
            # all are 1; break to avoid infinite
            break

    # if too few, add to buckets with remaining capacity
    while cur_sum() < limit:
        # pick split with max remaining capacity
        s = max(by_split.keys(), key=lambda x: len(by_split[x]) - alloc.get(x, 0))
        alloc[s] += 1

    picked: List[str] = []
    for s, ids in by_split.items():
        k_s = min(alloc.get(s, 0), len(ids))
        picked.extend(rnd.sample(ids, k_s))

    # adjust to exact limit if any mismatch (rare edge cases)
    if len(picked) > limit:
        picked = rnd.sample(picked, limit)
    elif len(picked) < limit:
        remaining = [x for x in all_ids if x not in set(picked)]
        picked.extend(rnd.sample(remaining, limit - len(picked)))

    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", required=True, help="docci_descriptions.jsonlines")
    ap.add_argument("--metadata", default=None, help="docci_metadata.jsonlines (optional)")
    ap.add_argument("--images-root", required=True, help="Folder containing extracted images")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")

    # subset control
    ap.add_argument("--limit", type=int, default=None, help="If set, only write N samples (e.g., 1000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed used when --limit is set")
    ap.add_argument("--subset-tag", default=None, help="Write subset_tag field (e.g., _1k)")
    ap.add_argument("--stratify-by-split", action="store_true", help="Sample proportionally by split")

    # output format
    ap.add_argument("--minimal", action="store_true",
                    help="If set, output only {id, split, image_path, text, subset_tag}")
    ap.add_argument("--keep-cloud-vision", action="store_true", help="Keep cloud_vision_api_responses (very large)")
    ap.add_argument("--keep-dsg", action="store_true", help="Keep dsg (large)")
    args = ap.parse_args()

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    img_map = index_images(args.images_root)
    if not img_map:
        raise RuntimeError(f"No images found under: {args.images_root}")

    desc_by_id = build_description_index(args.descriptions)
    meta_by_id = build_metadata_index(args.metadata, args.keep_cloud_vision, args.keep_dsg)

    # build valid id list (must have image on disk)
    valid_ids: List[str] = []
    id_to_split: Dict[str, str] = {}

    missing_img = 0
    for ex_id, d in desc_by_id.items():
        image_file = d["image_file"]
        if image_file not in img_map:
            missing_img += 1
            continue
        valid_ids.append(ex_id)
        id_to_split[ex_id] = d.get("split", "unknown") if d.get("split", None) is not None else "unknown"

    chosen_ids = choose_ids(
        all_ids=valid_ids,
        id_to_split=id_to_split,
        limit=args.limit,
        seed=args.seed,
        stratify_by_split=args.stratify_by_split,
    )

    # write
    missing_meta = 0
    written = 0

    for ex_id in chosen_ids:
        d = desc_by_id[ex_id]
        image_file = d["image_file"]
        text = d["description"]
        split = d.get("split", "unknown")

        img_path = img_map.get(image_file)
        if not img_path:
            # Shouldn't happen because we filtered valid_ids, but keep safe.
            continue

        subset_tag = args.subset_tag
        meta = meta_by_id.get(ex_id) if args.metadata else None
        if args.metadata and meta is None:
            missing_meta += 1

        if args.minimal:
            out = {
                "id": ex_id,
                "split": split,
                "image_path": os.path.abspath(img_path),
                "text": text,
            }
            if subset_tag is not None:
                out["subset_tag"] = subset_tag
        else:
            out = {
                "id": ex_id,
                "split": split,
                "image_file": image_file,
                "image_path": os.path.abspath(img_path),
                "text": text,
                "metadata": meta,
            }
            if subset_tag is not None:
                out["subset_tag"] = subset_tag

        write_jsonl_line(args.output, out)
        written += 1

    # summary
    print(f"Done.")
    print(f"descriptions_indexed={len(desc_by_id)}")
    print(f"valid_with_image={len(valid_ids)} (missing_img={missing_img})")
    if args.limit is not None:
        print(f"chosen={len(chosen_ids)} (limit={args.limit}, seed={args.seed}, stratify={args.stratify_by_split})")
    print(f"written={written}, missing_meta={missing_meta}")
    print(f"Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
