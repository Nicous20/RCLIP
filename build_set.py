#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DOCCI-style retrieval contrast captions from (image + original text).

Input JSONL (default): /workspace/clip/docci_pairs_10.jsonl
Expected fields per line:
  - id (str)
  - image_path (str)
  - text (str)   # original reference text

Output JSONL per sample:
  {
    "id": ...,
    "image_path": ...,
    "text": ...,
    "version": "v1|v2|v3",
    "sets": [ ... ]   # 5 tags; each has 1 gt + 4 neg
  }

Usage:
  # V1
  python build_set.py --version v1 --output /workspace/rclip/rclip_v1.jsonl --resume    

  # V2
  python build_set.py --version v2 --output /workspace/rclip/rclip_v2.jsonl --overwrite

  # V3
  python build_set.py --version v3 --output /workspace/rclip/rclip_v3.jsonl --resume
"""

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm
from google import genai
from google.genai import types


# -----------------------------
# JSONL IO
# -----------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    raise ValueError("Each line must be a JSON object.")
                yield obj
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_existing_ids(path: str, id_field: str = "id") -> Set[str]:
    if not os.path.exists(path):
        return set()
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and isinstance(obj.get(id_field), str):
                    ids.add(obj[id_field])
            except Exception:
                continue
    return ids


def count_nonempty_lines(path: str, max_scan: Optional[int] = None) -> int:
    """Count non-empty lines for tqdm total. If file is huge, you can set max_scan to None or use --no-count."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if max_scan is not None and i > max_scan:
                # fallback: unknown total
                return 0
            if line.strip():
                n += 1
    return n


# -----------------------------
# Gemini client
# -----------------------------
def make_client(api_key: Optional[str], timeout_ms: int) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing API key: set GEMINI_API_KEY/GOOGLE_API_KEY or pass --api-key")

    # Uses env proxy vars (http_proxy/https_proxy) automatically via httpx;
    # disable http2 for stability
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            api_version="v1beta",
            timeout=timeout_ms,
            client_args={"http2": False},
            async_client_args={"http2": False},
        ),
    )


def guess_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


# -----------------------------
# Robust JSON extraction
# -----------------------------
def extract_json_text(s: str) -> str:
    """Extract JSON object text from model output (handles ```json ...``` and extra chatter)."""
    s = (s or "").strip()

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()

    return s


def parse_json_from_model(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(extract_json_text(text))
    except Exception:
        return None


# -----------------------------
# Prompts
# -----------------------------
SYSTEM = "You are a dataset writer for perception-grounded reasoning retrieval. Follow the JSON format exactly. English only."

V1_TAGS = ["SUBJ_FORM", "NOUN_SWAP", "REL_PHRASE", "ATTR_STATE", "SENT_STRUCT"]
V2_TAGS = ["S", "A", "H", "T", "P"]

PROMPT_V1 = """You are given an image and a reference caption (for reference only; do NOT copy it verbatim):
{docci_description}

Create V1 (non-reasoning) retrieval contrast captions.

Output ONLY a JSON object with this exact structure:
{{
  "sets": [
    {{"tag":"SUBJ_FORM","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"NOUN_SWAP","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"REL_PHRASE","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"ATTR_STATE","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"SENT_STRUCT","gt":"...","neg":["...","...","...","..."]}}
  ]
}}

Global rules:
- English only. One sentence per caption.
- Each GT must be 30–40 tokens.
- Do NOT introduce new entities not visible.
- Each GT should include ONE mild, common-sense inference grounded in visible cues (use "likely / may indicate / suggesting").
  Examples: wear/rust -> likely older/poorly maintained; empty scene -> likely unattended; tidy placement -> likely parked.
- Each NEG must change exactly ONE key fact and must NOT add new entities.
- For each tag, the 4 NEGs must use 4 different error types.

Tag definitions for GT writing:
1) SUBJ_FORM: vary how you refer to the main subject (car/vehicle; definite vs descriptive NP), facts unchanged.
2) NOUN_SWAP: controlled noun/term substitutions (synonym / more specific / more general) without changing meaning.
3) REL_PHRASE: rewrite spatial/relational phrases while preserving the true relations.
4) ATTR_STATE: rewrite visible attributes/states in different wording without changing their values.
5) SENT_STRUCT: rewrite sentence structure/order without changing meaning.

NEG guidance (the single wrong fact must match the tag’s focus):
- SUBJ_FORM NEGs: flip ONE identity-related fact about the same main subject (e.g., color or count ONLY if clear). No new subjects.
- NOUN_SWAP NEGs: swap ONE noun to an incorrect but confusable alternative among ALREADY PRESENT elements (curb vs sidewalk; mural vs plain wall; etc.). Do NOT add unseen objects.
- REL_PHRASE NEGs: flip ONE spatial relation (left/right, front/back, in front/behind, on-street/on-sidewalk, near/far, touching/not touching).
- ATTR_STATE NEGs: flip ONE visible attribute/state (wall color; presence/absence of hubcaps; etc.).
- SENT_STRUCT NEGs: keep structure style but flip ONE of: orientation, location, or ONE attribute.

Return ONLY the JSON object. No extra text.
"""

PROMPT_V2 = """You are given an image and a reference caption (for reference only; do NOT copy it verbatim):
{docci_description}

Create V2 (perception-grounded reasoning) retrieval-contrast captions.

CRITICAL OUTPUT RULES:
- Output ONLY MINIFIED JSON (single line). No code fences. No explanations.
- Output must be a JSON object with EXACTLY this structure:
{{"sets":[{{"tag":"S","gt":"...","neg":["...","...","...","..."]}},{{"tag":"A","gt":"...","neg":["...","...","...","..."]}},{{"tag":"H","gt":"...","neg":["...","...","...","..."]}},{{"tag":"T","gt":"...","neg":["...","...","...","..."]}},{{"tag":"P","gt":"...","neg":["...","...","...","..."]}}]}}

GLOBAL CONTENT RULES (apply to ALL tags, GT and NEG):
1) English only. EXACTLY 2 sentences per caption.
2) Length: EACH caption must be 30–40 tokens (GT and every NEG).
3) Grounding:
   - Sentence 1 = ONLY directly visible, verifiable observations from the image.
   - Sentence 2 = EXACTLY ONE inference that is STRICTLY grounded in those visible cues.
   - Use one hedge word in sentence 2 ONLY: "suggesting" OR "likely" OR "may indicate" OR "could imply".
4) Entity constraint:
   - Do NOT introduce new entities not visible in the image. (No new people/animals/objects/brands/places.)
   - You may mention a category-level setting only if it is visually supported (e.g., "street", "kitchen", "gym floor").
5) NO ungrounded speculation:
   - Do NOT claim causes/events that are not visually evidenced (e.g., "car crash") unless there is clear visible damage consistent with it.
   - Avoid unverifiable intent or narrative.
6) NEG quality constraints (MANDATORY):
   - Each NEG must be a minimally edited rewrite of its GT: keep wording and structure as similar as possible.
   - Do NOT use contrastive/meta phrasing like "instead of", "rather than", "not ... but ...", or explanations.
   - Change EXACTLY ONE key factual span relative to GT; everything else should remain the same.
   - The changed fact MUST be objectively checkable from the image (not ambiguous, not viewpoint-dependent).
   - If a fact cannot be confirmed from the image with high confidence, do NOT use it for NEG.

TAG DEFINITIONS (for what sentence 1 should emphasize):
S: spatial/geometric relations (left/right, above/below, in front/behind, inside/outside, touching/not touching, parallel/perpendicular, near/far IF clearly near).
A: attributes/states (color, material, open/closed, on/off, present/absent parts, intact/broken, wet/dry, clean/dirty, rusty/new).
H: human/animal action; if none visible, explicitly state "No person or animal is visible." (verifiable).
T: temporal/phase ONLY if directly implied by visible evidence (day/night via lighting, shadows, artificial lights, rain/snow). Otherwise state: "No temporal phase is directly indicated in this single image."
P: physical intuition using visible cues only (support/contact, stability, load-bearing, resting/leaning, stacked/balanced).

NEG ERROR-TYPE TEMPLATES (each tag must use ALL 4 exactly once, each MUST be visually checkable):
S NEG types:
  S1) flip one relative position term (left/right OR above/below OR in front/behind) using the same objects.
  S2) flip one contact/containment relation (touching vs not touching OR inside vs outside OR on vs under).
  S3) flip one orientation relation if visible (parallel vs perpendicular OR facing direction if clear).
  S4) flip one clear proximity relation (near vs far) ONLY if the image clearly shows near; otherwise use another spatial flip that is clearly checkable.

A NEG types:
  A1) flip ONE color/material word that is clearly visible.
  A2) flip ONE part-presence/state (present vs missing; open vs closed; intact vs broken) that is clearly visible.
  A3) flip ONE surface condition (rusty vs clean; wet vs dry; dusty vs polished) only if visible.
  A4) flip ONE count/quantity attribute ONLY if the count is clearly visible; otherwise flip another visible attribute.

H NEG types (use only entities visible in the image; NO new entities):
  H1) flip presence: "No person/animal is visible" ↔ "A person/animal is visible" ONLY if that entity is actually visible; otherwise do NOT use this type.
  H2) flip occupancy/action of an already visible entity (e.g., "standing" ↔ "sitting", "holding" ↔ "not holding") ONLY if the action is visible.
  H3) flip motion vs stillness ONLY if motion cues are visible (blur, pose, interaction); otherwise do NOT use this type.
  H4) flip the actor-object relation ONLY if both actor and object are visible (e.g., "holding X" ↔ "holding Y") without introducing new entities.

IMPORTANT for H:
- If there is truly no visible person/animal, then ALL H NEGs must remain within the same visible-entity set and must still be checkable (e.g., incorrect claim that a visible statue is moving is NOT allowed; it is not a visible action). Prefer: claim a person is visible ONLY if a person is actually visible; otherwise H should focus on "no person/animal visible" and NEGs must flip within verifiable statements about visibility in-frame vs not (only if ambiguous? avoid). If H cannot produce 4 checkable NEGs without new entities, then make H sentence 1 about a visible human-made object/action evidence only when present (e.g., "a hand/tool is visible"), otherwise keep strictly "No person or animal is visible." and create NEGs by flipping visibility ONLY if a person/animal is actually present. If none are present, then produce H NEGs by flipping a clearly visible agent-like entity category already present (e.g., "a carved statue" is NOT a person/animal; do NOT claim it is). In this case, do NOT fabricate H NEGs; instead make sentence 1 describe the absence, and sentence 2 infer the scene is unoccupied, then NEGs flip ONE checkable environmental cue (e.g., lighting/day-night) is NOT allowed under H. Therefore: if no person/animal is visible, H NEGs must be about visibility within the frame ONLY (e.g., "No person is visible" ↔ "A person is visible") is NOT allowed unless person is present. Conclusion: If no person/animal visible, then H GT should state absence and inference; H NEGs must still avoid new entities—so they should flip ONLY an already visible human/animal if any. If none, keep H identical to GT for all negs is NOT allowed. Therefore, only create H NEGs when there is a visible person or animal. If none, set H gt to: "No person or animal is visible." plus inference, and set all 4 negs to be the same length but each changes ONE verifiable fact about other visible cues? NO. This violates tag focus. So: If no person/animal visible, then H must still be grounded: sentence 1 about absence, sentence 2 about unoccupied; and H NEGs must flip ONE checkable statement about visibility of people/animals ONLY if any are visible. If none are visible, then set H NEGs to claim different specific visible people/animals is NOT allowed. Hence: ONLY produce H when a person or animal is visible; otherwise set H gt and all negs to empty strings is NOT allowed. To avoid this, assume typical images may have no people; you MUST still produce H with 4 NEGs that are checkable WITHOUT adding entities: Use visibility of body parts IF visible (hands/legs) counts as person. If no body parts, then use "reflection" or "shadow of a person" ONLY if clearly visible. If absolutely no human/animal evidence, then H sentence 1 should still be "No person or animal is visible." and NEGs should flip the claim to "A person is visible" ONLY if a person is actually visible; otherwise DO NOT do that. In that case, use H NEGs by flipping the subject from "person or animal" to "animal" vs "person" is not checkable. Therefore, if no evidence, you MUST make H about human/animal absence and set NEGs by flipping ONE clearly visible cue about potential presence: e.g., "an empty chair" does not imply person. This is tricky: the safest is to treat "H" as "agent presence" and only flip if there is evidence. So follow this rule:
  - If no human/animal evidence is visible, set H GT to absence; for H NEGs, flip ONLY the presence of human/animal evidence that IS visible (e.g., a visible hand, a visible reflection, a visible shadow). If none exists, then you must still create 4 NEGs by flipping a DIFFERENT, clearly visible detail in sentence 1 while keeping the tag "H" and sentence 2 inference about occupancy. These NEGs must remain plausible and checkable, but the changed fact must still be in sentence 1. (Example: change where an object is, or an attribute) BUT keep sentence 2 about unoccupied. This is allowed as long as you do not introduce entities and keep the overall H framing.
(Yes, do this fallback if needed.)

T NEG types:
  T1) day ↔ night based on lighting/shadows ONLY if clearly indicated.
  T2) claim artificial lighting ↔ natural lighting ONLY if clearly indicated.
  T3) claim weather condition (rain/snow) ONLY if visible; otherwise do NOT use.
  T4) flip "No temporal phase is directly indicated" ↔ a specific temporal claim ONLY if evidence supports the specific claim; otherwise keep "no temporal phase" and flip a different checkable temporal cue (e.g., shadow direction) only if visible.

P NEG types:
  P1) flip contact: resting on vs hovering above (only if contact is visible).
  P2) flip stability: stable vs unstable/tilting ONLY if tilt is visible.
  P3) flip support source: supported by surface vs supported by another object ONLY if both are visible.
  P4) flip load-bearing/stacking relation ONLY if stacking is visible; otherwise use another contact flip.

FINAL CHECK BEFORE OUTPUT:
- For every NEG: the wrong fact must be CLEARLY contradicted by visible evidence in the image (not ambiguous).
- Keep GT and NEG captions stylistically identical and similarly long.
- Output only the JSON object, minified, single line.
"""

PROMPT_V3 = """You are given an image and a reference caption (for reference only; do NOT copy it verbatim):
{docci_description}

Create V3 (enhanced-reasoning) retrieval contrast captions.

IMPORTANT:
- Think step-by-step internally, BUT DO NOT output chain-of-thought or multi-step reasoning.
- Instead, each GT caption must include ONE short, evidence-grounded inference using "suggesting/indicating/implying/because/so".
- The inference MUST be directly supported by visible evidence. If uncertain, do NOT infer; be conservative.
- Output ONLY the final JSON. Return MINIFIED JSON (single line). Do NOT use code fences.

Output ONLY a JSON object with this exact structure:
{{
  "sets": [
    {{"tag":"S","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"A","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"H","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"T","gt":"...","neg":["...","...","...","..."]}},
    {{"tag":"P","gt":"...","neg":["...","...","...","..."]}}
  ]
}}

Global rules:
- English only. One sentence per caption.
- Each GT must be 30–40 tokens.
- Do NOT introduce new entities not visible.
- Each GT MUST contain: observation + a stronger inference, and MAY add ONE alternative hypothesis using "or possibly".
  Example form: "... suggesting X, or possibly Y."
- Inferences MUST still be grounded in visible cues; avoid pure fantasy or unseen events.
- Each NEG changes exactly ONE key fact, no new entities, plausible caption.
- For each tag, 4 NEGs must use 4 different error types.
- Each GT = concrete observation + ONE mild inference grounded by a visible cue (single clause).
- Do NOT infer hidden intent/identity/unseen events.

Tag-specific GT focus:
S: spatial/geometric relations + one evidence-grounded inference about depth/layout.
A: attributes/states + one evidence-grounded inference about condition (e.g., glossy suggests wet).
H: action if visible; otherwise explicitly say none is present, in reasoning style.
T: only if visually implied; otherwise explicitly state no temporal phase is directly indicated, in reasoning style.
P: physical intuition + one inference about support/stability from contact cues.

NEG rules per tag:
S NEGs: flip one spatial relation.
A NEGs: flip one visible attribute/state.
H NEGs: flip presence/action ONLY using entities that are visible.
T NEGs: flip temporal claim (or flip "no temporal phase indicated").
P NEGs: flip one physical relation.

Return ONLY the JSON object. No extra text.
"""


def get_prompt_and_tags(version: str) -> Tuple[str, List[str]]:
    v = version.lower()
    if v == "v1":
        return PROMPT_V1, V1_TAGS
    if v == "v2":
        return PROMPT_V2, V2_TAGS
    if v == "v3":
        return PROMPT_V3, V2_TAGS
    raise ValueError(f"Unknown --version {version}. Choose from v1, v2, v3.")


# -----------------------------
# Validation
# -----------------------------
def validate_sets_obj(obj: Dict[str, Any], expect_tags: List[str]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a json object"
    sets = obj.get("sets")
    if not isinstance(sets, list) or len(sets) != 5:
        return "sets must be list length 5"

    seen = set()
    for it in sets:
        if not isinstance(it, dict):
            return "each sets item must be an object"
        tag = it.get("tag")
        gt = it.get("gt")
        neg = it.get("neg")
        if tag not in expect_tags:
            return f"invalid tag: {tag}"
        if tag in seen:
            return f"duplicate tag: {tag}"
        seen.add(tag)
        if not isinstance(gt, str) or not gt.strip():
            return f"empty gt for tag={tag}"
        if not isinstance(neg, list) or len(neg) != 4:
            return f"neg must be length-4 list for tag={tag}"
        for j, n in enumerate(neg):
            if not isinstance(n, str) or not n.strip():
                return f"empty neg[{j}] for tag={tag}"

    if set(seen) != set(expect_tags):
        return f"tags mismatch: got {sorted(seen)}"
    return None


# -----------------------------
# Gemini call + outer retry (handles truncation / missing tags)
# -----------------------------
def call_gemini_once(
    client: genai.Client,
    model: str,
    image_bytes: bytes,
    image_mime: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
) -> str:
    cfg = types.GenerateContentConfig(
        system_instruction=[SYSTEM],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
            prompt,
        ],
        config=cfg,
    )
    return (resp.text or "").strip()


def generate_with_retry(
    client: genai.Client,
    model: str,
    image_bytes: bytes,
    image_mime: str,
    prompt: str,
    expect_tags: List[str],
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    retries: int,
    sleep_s: float,
) -> Tuple[Dict[str, Any], str]:
    """
    Try to get a complete JSON with all tags. If truncated / parse fail / missing tags, re-ask with repair suffix.
    Returns: (parsed_obj, raw_text)
    """
    repair_suffix = """
IMPORTANT: Your previous output may be truncated or incomplete.
Re-output the FULL JSON with ALL FIVE tags.
Return MINIFIED JSON on a single line.
Do NOT use code fences.
Return ONLY the JSON object.
"""

    last_raw = ""
    for attempt in range(retries + 1):
        try:
            raw = call_gemini_once(
                client=client,
                model=model,
                image_bytes=image_bytes,
                image_mime=image_mime,
                prompt=prompt + (repair_suffix if attempt > 0 else ""),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            last_raw = raw
            parsed = parse_json_from_model(raw)
            if parsed is None:
                raise ValueError("parse failed (non-JSON or truncated)")
            err = validate_sets_obj(parsed, expect_tags)
            if err is None:
                return parsed, raw
            raise ValueError(f"format invalid: {err}")
        except Exception:
            if attempt < retries:
                backoff = sleep_s * (2 ** attempt)
                backoff *= (0.7 + 0.6 * random.random())
                time.sleep(backoff)
            else:
                break

    raise RuntimeError("Failed after retries. Last raw head:\n" + (last_raw[:600] if last_raw else "<empty>"))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Build DOCCI sets for V1/V2/V3 from (image + text).")
    ap.add_argument("--version", required=True, choices=["v1", "v2", "v3"], help="Which dataset version to generate.")
    ap.add_argument("--input", default="/workspace/rclip/docci_pairs_3.jsonl", help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")

    ap.add_argument("--id-field", default="id")
    ap.add_argument("--image-field", default="image_path")
    ap.add_argument("--text-field", default="text")

    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--api-key", default=None)

    ap.add_argument("--timeout-ms", type=int, default=180000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=100000)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)

    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.8)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-count", action="store_true", help="Do not pre-count lines for tqdm total (faster start).")

    args = ap.parse_args()

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    done = load_existing_ids(args.output) if args.resume else set()

    prompt_tpl, tags = get_prompt_and_tags(args.version)
    client = make_client(args.api_key, args.timeout_ms)

    total = 0 if args.no_count else count_nonempty_lines(args.input, max_scan=None)
    iterator = read_jsonl(args.input)
    pbar = tqdm(iterator, total=total if total > 0 else None, dynamic_ncols=True)

    for item in pbar:
        sid = item.get(args.id_field)
        img_path = item.get(args.image_field)
        txt = item.get(args.text_field)

        if not isinstance(sid, str) or not sid:
            continue
        if args.resume and sid in done:
            continue

        # basic checks
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            append_jsonl(args.output, {"id": sid, "image_path": img_path, "error": f"missing image: {img_path}"})
            done.add(sid)
            continue
        if not isinstance(txt, str) or not txt.strip():
            append_jsonl(args.output, {"id": sid, "image_path": img_path, "error": "missing text field"})
            done.add(sid)
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        mime = guess_mime_type(img_path)

        prompt = prompt_tpl.format(docci_description=txt.strip())

        try:
            parsed, raw = generate_with_retry(
                client=client,
                model=args.model,
                image_bytes=img_bytes,
                image_mime=mime,
                prompt=prompt,
                expect_tags=tags,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                retries=args.retries,
                sleep_s=args.sleep,
            )

            out_obj = {
                "id": sid,
                "image_path": img_path,
                "text": txt.strip(),
                "version": args.version,
                "sets": parsed["sets"],
                # Uncomment if you want to keep raw for debugging cost/truncation:
                # "gemini_raw": raw,
            }
            append_jsonl(args.output, out_obj)
            done.add(sid)

        except Exception as e:
            append_jsonl(args.output, {"id": sid, "image_path": img_path, "error": str(e)})
            done.add(sid)

    try:
        client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
