#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry failed / missing items from a previous DOCCI generation run.

Three categories are handled (all keyed by the "id" field, NOT line number):

  1. Entries in --failed-input with NO "error" field  →  written directly to output (pass-through).
  2. Entries in --failed-input WITH an "error" field  →  re-generated using original data.
  3. IDs present in --original-input but ABSENT from --failed-input entirely  →  generated fresh.

Categories 2 + 3 are processed together in a multi-threaded pool.
The final output contains exactly one entry per ID in --original-input.

Usage:
  python retry_failed.py --version v2 \
    --original-input /workspace/RCLIP/docci_pairs_5k.jsonl \
    --failed-input   /workspace/RCLIP/rclip_5k_v1_gpt.jsonl \
    --output         /workspace/RCLIP/rclip_5k_v1_gpt_final.jsonl \
    --model gpt-4.1-mini \
    --workers 8 \
    --qps 4.0

Environment:
  export OPENAI_API_KEY=...
"""

import argparse
import base64
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm
from openai import OpenAI


# ──────────────────────────────────────────────
# JSONL helpers
# ──────────────────────────────────────────────

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    raise ValueError("not a JSON object")
                yield obj
            except Exception as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e


def write_line(path: str, obj: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


# ──────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────

def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"


def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{guess_mime(path)};base64,{b64}"


# ──────────────────────────────────────────────
# JSON extraction / parsing
# ──────────────────────────────────────────────

def extract_json(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        return s[i:j + 1].strip()
    return s


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(extract_json(text))
    except Exception:
        return None


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

SYSTEM = "You are a dataset writer for perception-grounded reasoning retrieval. Follow the JSON format exactly. English only."

V1_TAGS = ["SUBJ_FORM", "NOUN_SWAP", "REL_PHRASE", "ATTR_STATE", "SENT_STRUCT"]
V2_TAGS = ["S", "A", "H", "T", "P"]

# TODO: paste your latest prompts here
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

Create V3 (inference-only contrast) retrieval-contrast captions.

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

V3 KEY DIFFERENCE (INFERENCE-ONLY NEGATIVES):
6) NEG quality constraints (MANDATORY):
   - Each NEG must be a minimally edited rewrite of its GT: keep wording and structure as similar as possible.
   - Do NOT use contrastive/meta phrasing like "instead of", "rather than", "not ... but ...", or explanations.
   - Sentence 1 MUST remain IDENTICAL to the GT (character-for-character, except spaces/punctuation normalization).
   - The ONLY change allowed is in Sentence 2, and it MUST be inside the inference clause.
   - Change EXACTLY ONE key factual span in the inference clause relative to GT; everything else should remain the same.
   - The changed inference must still be plausible given sentence 1, but be objectively WRONG or less-supported when compared to the image evidence.
   - If an inference span cannot be made clearly wrong (or clearly less-supported) from the image with high confidence, do NOT use it.

TAG DEFINITIONS (for what sentence 1 should emphasize):
S: spatial/geometric relations (left/right, above/below, in front/behind, inside/outside, touching/not touching, parallel/perpendicular, near/far IF clearly near).
A: attributes/states (color, material, open/closed, on/off, present/absent parts, intact/broken, wet/dry, clean/dirty, rusty/new).
H: human/animal action; if none visible, explicitly state "No person or animal is visible." (verifiable).
T: temporal/phase ONLY if directly implied by visible evidence (day/night via lighting, shadows, artificial lights, rain/snow). Otherwise state: "No temporal phase is directly indicated in this single image."
P: physical intuition using visible cues only (support/contact, stability, load-bearing, resting/leaning, stacked/balanced).

NEG ERROR-TYPE TEMPLATES (each tag must use ALL 4 exactly once, applied ONLY to Sentence 2 inference clause):
S inference NEG types:
  S1) flip inferred layout/depth relation (nearer vs farther; foreground vs background) ONLY if strongly supported by cues.
  S2) flip inferred placement intention (aligned/organized vs scattered) ONLY if supported by arrangement cues.
  S3) flip inferred viewpoint effect (close-up vs wide/room-scale) ONLY if supported by framing cues.
  S4) flip inferred functional spatial implication (blocking passage vs leaving clearance) ONLY if supported by clear space cues.

A inference NEG types:
  A1) flip inferred condition (well-maintained vs worn) grounded in visible wear/cleanliness cues.
  A2) flip inferred material implication (heavy vs light) ONLY if supported by visible material/structure cues.
  A3) flip inferred usage state (recently used vs unused) ONLY if supported by visible cues (heat/steam/wetness/mess).
  A4) flip inferred freshness/age (new vs old) ONLY if supported by visible cues (rust, peeling, scratches).

H inference NEG types:
  H1) flip inferred occupancy (recently occupied vs unoccupied) ONLY if supported by strong cues (visible person/body part/shadow/reflection OR clearly used items).
  H2) flip inferred activity level (active use vs stored) ONLY if supported by pose/placement cues.
  H3) flip inferred interaction (someone just handled it vs untouched) ONLY if supported by fingerprints/mess/positioning cues.
  H4) flip inferred human presence cue (presence implied vs not implied) ONLY if supported by shadows/reflections/body parts; otherwise choose another H inference flip that remains defensible.

T inference NEG types:
  T1) flip inferred time-of-day (day vs night) ONLY if strongly supported by lighting cues.
  T2) flip inferred lighting source (natural vs artificial) ONLY if strongly supported.
  T3) flip inferred weather recency (recent rain vs dry) ONLY if supported by wet surfaces/puddles.
  T4) flip inferred temporality of activity (ongoing vs paused) ONLY if supported by cues (open tools, active displays).

P inference NEG types:
  P1) flip inferred stability (stable vs precarious) ONLY if supported by visible balance/contact.
  P2) flip inferred support/load (bearing weight vs not) ONLY if supported by stacking/contact.
  P3) flip inferred mobility (fixed in place vs easily movable) ONLY if supported by size/material cues.
  P4) flip inferred contact certainty (firmly resting vs barely touching) ONLY if supported by clear contact cues.

FINAL CHECK BEFORE OUTPUT:
- Sentence 1 is identical across GT and all 4 NEGs within the same set.
- For every NEG, ONLY the inference clause in sentence 2 changes, and only ONE key span changes.
- The wrong inference must be contradicted or strongly less-supported by visible evidence (not ambiguous).
- Keep GT and NEG captions stylistically identical and similarly long.
- Output only the JSON object, minified, single line.
"""


def get_prompt_and_tags(version: str) -> Tuple[str, List[str]]:
    v = version.lower()
    if v == "v1":
        return PROMPT_V1, V1_TAGS
    if v == "v2":
        return PROMPT_V2, V2_TAGS
    if v == "v3":
        return PROMPT_V3, V2_TAGS
    raise ValueError(f"Unknown version: {version}")


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_sets(obj: Dict[str, Any], expect_tags: List[str]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a dict"
    sets = obj.get("sets")
    if not isinstance(sets, list) or len(sets) != 5:
        return "sets must be list of length 5"
    seen: Set[str] = set()
    for it in sets:
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


# ──────────────────────────────────────────────
# QPS limiter
# ──────────────────────────────────────────────

class QPSLimiter:
    def __init__(self, qps: float):
        self.interval = 1.0 / max(qps, 1e-9)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.time()
            self.next_time = now + self.interval


# ──────────────────────────────────────────────
# OpenAI call
# ──────────────────────────────────────────────

def call_once(client: OpenAI, model: str, image_url: str,
              prompt: str, max_tokens: int, temperature: float) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": prompt},
            ]},
        ],
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    out = getattr(resp, "output_text", None)
    if isinstance(out, str) and out.strip():
        return out.strip()
    try:
        chunks = []
        for item in resp.output:
            for c in item.content:
                if c.type in ("output_text", "text"):
                    chunks.append(getattr(c, "text", ""))
        return "".join(chunks).strip()
    except Exception:
        return ""


REPAIR_SUFFIX = """
IMPORTANT: Your previous output may be truncated or incomplete.
Re-output the FULL JSON with ALL FIVE tags.
Return MINIFIED JSON on a single line. No code fences. JSON only.
"""


def generate(client: OpenAI, model: str, image_url: str, prompt: str,
             tags: List[str], max_tokens: int, temperature: float,
             retries: int, base_sleep: float,
             limiter: Optional[QPSLimiter]) -> Dict[str, Any]:
    last_raw = ""
    for attempt in range(retries + 1):
        try:
            if limiter:
                limiter.wait()
            raw = call_once(client, model, image_url,
                            prompt + (REPAIR_SUFFIX if attempt > 0 else ""),
                            max_tokens, temperature)
            last_raw = raw
            parsed = parse_model_json(raw)
            if parsed is None:
                raise ValueError("parse failed")
            err = validate_sets(parsed, tags)
            if err is None:
                return parsed
            raise ValueError(f"format invalid: {err}")
        except Exception:
            if attempt < retries:
                sleep = base_sleep * (2 ** attempt) * (0.7 + 0.6 * random.random())
                time.sleep(sleep)
    head = last_raw[:600] if last_raw else "<empty>"
    raise RuntimeError(f"Failed after retries. Last output head:\n{head}")


# ──────────────────────────────────────────────
# Per-item worker
# ──────────────────────────────────────────────

def process_one(item: Dict[str, Any], args: argparse.Namespace,
                prompt_tpl: str, tags: List[str],
                client: OpenAI, limiter: Optional[QPSLimiter]) -> Dict[str, Any]:
    sid = item.get(args.id_field)
    img_path = item.get(args.image_field)
    txt = item.get(args.text_field)

    if not isinstance(sid, str) or not sid:
        return {"error": "missing id"}
    if not isinstance(img_path, str) or not os.path.exists(img_path):
        return {"id": sid, "image_path": img_path, "error": f"image not found: {img_path}"}
    if not isinstance(txt, str) or not txt.strip():
        return {"id": sid, "image_path": img_path, "error": "missing text"}

    image_url = to_data_url(img_path)
    prompt = prompt_tpl.format(docci_description=txt.strip())

    parsed = generate(client, args.model, image_url, prompt, tags,
                      args.max_output_tokens, args.temperature,
                      args.retries, args.sleep, limiter)
    return {
        "id": sid,
        "image_path": img_path,
        "text": txt.strip(),
        "version": args.version,
        "sets": parsed["sets"],
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Retry failed/missing DOCCI entries and merge with successes into one JSONL."
    )
    ap.add_argument("--version", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--original-input", required=True,
                    help="Original input JSONL with id / image_path / text for all items.")
    ap.add_argument("--failed-input", required=True,
                    help="Previous output JSONL (mix of successes and error entries).")
    ap.add_argument("--output", required=True,
                    help="New merged output JSONL (overwritten if exists).")

    ap.add_argument("--id-field",    default="id")
    ap.add_argument("--image-field", default="image_path")
    ap.add_argument("--text-field",  default="text")

    ap.add_argument("--model",            default="gpt-4.1-mini")
    ap.add_argument("--api-key",          default=None)
    ap.add_argument("--max-output-tokens", type=int,   default=2000)
    ap.add_argument("--temperature",       type=float, default=0.2)
    ap.add_argument("--retries",           type=int,   default=6)
    ap.add_argument("--sleep",             type=float, default=0.8)
    ap.add_argument("--workers",           type=int,   default=4)
    ap.add_argument("--qps",               type=float, default=0.0)

    args = ap.parse_args()

    # Always start fresh
    if os.path.exists(args.output):
        print(f"[warn] Overwriting existing output: {args.output}")
        os.remove(args.output)

    prompt_tpl, tags = get_prompt_and_tags(args.version)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key")
    client = OpenAI(api_key=api_key)

    limiter = QPSLimiter(args.qps) if args.qps > 0 else None
    lock = threading.Lock()

    # ── Step 1: load original input keyed by id ────────────────────────
    print(f"[1/3] Loading original-input: {args.original_input}")
    original: Dict[str, Dict[str, Any]] = {}
    for item in read_jsonl(args.original_input):
        sid = item.get(args.id_field)
        if isinstance(sid, str) and sid:
            original[sid] = item
    print(f"      {len(original)} unique IDs loaded.")

    # ── Step 2: scan failed-input, classify by id ──────────────────────
    print(f"[2/3] Scanning failed-input: {args.failed_input}")

    # ids that appear in failed-input (regardless of success/error)
    seen_ids: Set[str] = set()
    # ids that were successful in failed-input
    success_ids: Set[str] = set()

    n_passthrough = 0

    for entry in read_jsonl(args.failed_input):
        sid = entry.get(args.id_field)
        if not isinstance(sid, str) or not sid:
            continue  # skip malformed lines

        seen_ids.add(sid)

        if entry.get("error"):
            # failed entry — do NOT write to output yet, will be re-generated
            pass
        else:
            # successful entry — write directly to output
            success_ids.add(sid)
            write_line(args.output, entry, lock)
            n_passthrough += 1

    # ids to re-generate:
    #   (a) appeared in failed-input but had an error
    #   (b) never appeared in failed-input at all
    failed_ids  = seen_ids - success_ids                        # (a)
    missing_ids = set(original.keys()) - seen_ids               # (b)
    todo_ids    = failed_ids | missing_ids

    # build item list for generation (look up original data by id)
    items_to_generate: List[Dict[str, Any]] = []
    for sid in todo_ids:
        if sid in original:
            items_to_generate.append(original[sid])
        else:
            # id was in failed-input but not in original — keep error as-is
            write_line(args.output, {"id": sid, "error": "not found in original-input"}, lock)

    print(f"      Passed through (success):   {n_passthrough}")
    print(f"      To re-generate (error):     {len(failed_ids)}")
    print(f"      To generate (missing):      {len(missing_ids)}")
    print(f"      Total to generate:          {len(items_to_generate)}")
    print(f"      Expected output total:      {n_passthrough + len(items_to_generate)}")

    if not items_to_generate:
        print("\nNothing to generate. Output is complete.")
        return

    # ── Step 3: multi-threaded generation ─────────────────────────────
    print(f"\n[3/3] Generating {len(items_to_generate)} items "
          f"with {args.workers} worker(s)...")

    n_ok = 0
    n_fail = 0
    pbar = tqdm(total=len(items_to_generate), dynamic_ncols=True, desc="Generating")

    if args.workers <= 1:
        for item in items_to_generate:
            try:
                out = process_one(item, args, prompt_tpl, tags, client, limiter)
                n_ok += 1
            except Exception as e:
                out = {"id": item.get(args.id_field),
                       "image_path": item.get(args.image_field),
                       "error": str(e)}
                n_fail += 1
            write_line(args.output, out, lock)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(process_one, item, args, prompt_tpl, tags, client, limiter): item
                for item in items_to_generate
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    out = future.result()
                    n_ok += 1
                except Exception as e:
                    out = {"id": item.get(args.id_field),
                           "image_path": item.get(args.image_field),
                           "error": str(e)}
                    n_fail += 1
                write_line(args.output, out, lock)
                pbar.update(1)

    pbar.close()

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n=== Done ===")
    print(f"  Passed through (already OK):  {n_passthrough}")
    print(f"  Generated successfully:       {n_ok}")
    print(f"  Still failed after retry:     {n_fail}")
    print(f"  Total entries in output:      {n_passthrough + n_ok + n_fail}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()