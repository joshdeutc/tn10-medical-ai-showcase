"""Extract whether a central venous catheter is present in clinical notes.

The corpus arrives as a CSV with three columns — `id_note`, `input`, and `result`
("Y"/"N") if you want to evaluate against a reference. The script loads a model in
4-bit, runs every note through it, and writes two files: the per-note detail and one
line of metrics.

    python3 extraire.py --data notes.csv
    python3 extraire.py --data notes.csv --gold labels.csv --label "medgemma @ HR"

With neither `--gold` nor a `result` column, the extraction produces the predictions
without evaluating them — and `verify.csv` still reports hallucinations and format
failures, which require no ground truth.

The default window is 16384 rather than 8192: at 8192 Croatian would lose 7547 tokens
to truncation against 1135 for English, and the F1 gap would be measuring that
asymmetry instead of the quality of the text. At 16384 both languages pass through
whole (at most 6596 tokens in English, 7677 in Croatian).

This file is assembled, word for word, from the cells of the original notebook
(notebooks/extraction_local.ipynb, since removed): this exact code produced
results/tableau_final.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch      # the notebook imported it in cell 2, at module level: the functions
                  # copied below use it without importing it again

# Default values, overwritten by main() before any call. The cells copied below read
# them at module level (VERIFY_PATH derives from RESULTS_DIR), so they have to exist
# before those cells.
RESULTS_DIR = Path("results")
DEFAULT_MODEL = "unsloth/medgemma-27b-text-it-bnb-4bit"

def compute_number_of_tokens(full_text: str, tokenizer) -> int:
    """Number of tokens in a text."""
    tokens = tokenizer(full_text)["input_ids"]
    return len(tokens)


def compute_remaining_token_budget(
    tokenizer,
    max_seq_length: int,
    fixed_messages: list[dict],
    safety_margin: int,
) -> int:
    """Tokens left for the note, once the window has been reduced by the chat
    template (system + user without the note) and by the margin reserved for
    generation."""
    fixed_tokens = len(tokenizer.apply_chat_template(fixed_messages, add_generation_prompt=True))
    return max_seq_length - fixed_tokens - safety_margin


def truncate_text_to_token_budget(text: str, token_budget: int, tokenizer) -> str:
    """Cut a text so that it does not exceed token_budget tokens."""
    tokens = tokenizer(text)["input_ids"]
    if len(tokens) <= token_budget:
        return text
    return tokenizer.decode(tokens[:token_budget], skip_special_tokens=True)

SYSTEM_PROMPT = """# ROLE
You are a clinical data-extraction component in a bloodstream-infection (BSI)
surveillance pipeline. You are given the free-text discharge summary for a SINGLE
patient's SINGLE hospitalisation episode. Your task is to determine, strictly from what the text documents, whether the patient had a central venous catheter (CVC) during this episode. Your
output is scored against human expert coding: correct calibration matters more than
completeness, and asserting something the text does not support is worse than saying
it is absent.

# NOTE FORMAT
These are English hospital discharge summaries. Identifiers and most dates are
redacted and appear as "___". Treat "___" as MISSING, never as content: never read a
value out of a redaction and never reconstruct one. Central lines, when present, are
most often documented in the "Major Surgical or Invasive Procedure" field, in the
"Brief Hospital Course", or incidentally in imaging under "Pertinent Results" (e.g. a
chest X-ray noting an IJ or PICC tip). Scan all of these sections, not only the
procedure field.

# WHAT COUNTS AS A CVC
A CVC (central venous catheter) is a catheter whose tip sits in a large central vein
or the right heart, placed for venous access — drugs, fluids, parenteral nutrition,
haemodynamic monitoring, or dialysis/apheresis. The defining property is that the
device is CENTRAL and VENOUS. Route of insertion does not decide it: a PICC is
inserted in the arm but its tip is central, so it counts.

Counts as CVC = Yes:
- Central line / central venous catheter / CVC
- Non-tunnelled CVC — internal jugular (IJ), subclavian, or femoral line
- Tunnelled CVC — Hickman, Broviac, Groshong
- PICC — peripherally inserted central catheter (tip typically in the SVC)
- Implanted port — Port-a-Cath, mediport, "port"
- Dialysis / apheresis catheter — Vas-Cath, Quinton, Permcath, Shaldon,
  tunnelled dialysis catheter, "HD line"
- Pulmonary-artery (Swan-Ganz) catheter, or a large-bore central introducer / cordis

Does NOT count — code CVC = No, UNLESS a separate qualifying central line is also
documented:
- Urinary / bladder catheter — Foley, indwelling urinary catheter, straight cath /
  intermittent self-catheterization, suprapubic catheter, condom catheter. Drains the
  bladder; not a vascular device.
- Any stent — ureteral / ureteric / double-J (JJ), biliary, coronary, or
  peripheral-vascular stent. A stent holds a lumen open; it is not a venous catheter.
- Peripheral venous access — peripheral IV, PIV, "IV", saline lock, peripheral
  cannula.
- Midline catheter — tip stops short of a central vein; count only if the text
  explicitly states a central tip position.
- Arterial line — A-line, radial or femoral arterial line (arterial, not venous).
- Non-vascular tubes — nasogastric / OG tube, chest tube, surgical drains (e.g. JP
  drain), epidural catheter, PEG / feeding tube.

# DECISION RULES
1. DOCUMENTED, NOT INFERRED. Code Yes only if a central venous device is named or
   unambiguously described in the text. Do NOT infer a CVC from context alone: ICU
   admission, vasopressors, chemotherapy, transfusions, parenteral nutrition, major
   surgery, or a bare mention of "IV fluids/antibiotics" are NOT sufficient on their
   own. If it is not written, it is not counted.
2. THIS EPISODE ONLY. Ignore devices from prior admissions or from past
   medical/surgical history UNLESS the text shows the device continued into, or was in
   use during, this hospitalisation episode. Removal, culture, flushing or any other
   manipulation of the device during this admission IS that evidence: a line pulled or
   cultured during the stay was present during the stay. A missing or redacted
   insertion date is NOT a reason to code No — rule 4 prevails over this rule.
3. READ THE DEVICE, NOT THE WORD. The string "catheter" / "catheterization"
   frequently refers to the WRONG device (urinary catheter, cardiac catheterization,
   ureteral stent). Conversely, genuine CVCs are often named WITHOUT that string —
   "IJ", "PICC", "port", "Vas-Cath", "central line". Decide on device type and tip
   location, not on keyword presence.
4. A CVC CAN BE DOCUMENTED INDIRECTLY. It may appear only as it is placed
   ("Major Surgical or Invasive Procedure: R IJ placement"), only as it is removed or
   cultured ("the IJ was pulled and the catheter tip sent"), or incidentally on
   imaging ("right IJ catheter unchanged" on CXR; "PICC tip terminates in the upper
   SVC"). Any of these counts as Yes.
5. CARDIAC CATHETERIZATION IS NOT A CVC. It is a coronary angiogram via arterial
   access — a procedure, not an indwelling central venous line. Code it No on its own,
   but still scan the rest of the note for a genuine central line, which may be
   documented separately (e.g. an IJ on a post-op CXR after CABG).
6. HANDLE NEGATION. "no foley", "no central line", "line removed prior to admission,
   none since" document ABSENCE. Do not code Yes off a device word that is negated.
7. WHEN GENUINELY AMBIGUOUS, CODE No AND FLAG. An unqualified "line", "line flush", or
   "delined" with no named or described central device does NOT establish a CVC.
   Default such cases to CVC = No and set "uncertain": true for human review. Do not
   guess Yes. This default applies ONLY when no central venous device is named or
   described; when one is (rule 4), code Yes even if the timing is incomplete."""


def build_user_prompt(full_text: str) -> str:
    return f"""# OUTPUT
Return a SINGLE valid JSON object and NOTHING else — no prose, no markdown, no code
fences, no trailing text. Use exactly this schema and these keys:

{{
  "CVC": "Yes" | "No",
  "evidence": "",
  "uncertain": true | false
}}

Field rules:
- CVC: exactly "Yes" or "No".
- evidence: a verbatim span (or spans) copied from the source text. If CVC = "Yes",
  this MUST contain the phrase that names or describes the central venous device. If
  CVC = "No", quote the most relevant span (the negation, the non-central device, or
  the ambiguous phrase), or use "no central venous device documented". Never fabricate
  text that is not in the source.
- uncertain: true when central-line status could not be resolved with confidence
  (rule 7); false otherwise.
- Be literal and consistent. Do not speculate beyond the text. Do not use outside
clinical knowledge to fill gaps. Identical inputs must produce identical outputs.

Report :
---
{full_text}
---"""

import gc, time, psutil

MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 512          # headroom for "thinking" models (Qwen3)
LOAD_IN_4BIT = True           # mandatory for the 27B/32B/70B on 40 GB

# VRAM of the TARGET machine: acts as a warning when preparing a run for an
# A100 40 GB from a machine that has more. Set to None to use the VRAM
# actually available.
VRAM_BUDGET_GB = 40

from unsloth import FastLanguageModel

model, tokenizer = None, None

def free_memory():
    """Free the previous model's VRAM and RAM — the del -> gc -> empty_cache order matters."""
    global model, tokenizer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    model, tokenizer = None, None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def report_memory(prefix=""):
    ram = psutil.virtual_memory()
    vram_free, vram_total = torch.cuda.mem_get_info()
    print(f"  {prefix}RAM free: {ram.available/1e9:.1f}/{ram.total/1e9:.1f} GB"
          f" | VRAM free: {vram_free/1e9:.1f}/{vram_total/1e9:.1f} GB")


def estimate_kv_cache_gb(seq_len: int = None, batch_size: int = 1) -> float:
    """KV cache size at full window, read from the model config.

    2 (K and V) × layers × kv_heads × head_dim × seq_len × 2 bytes (bf16).
    `num_key_value_heads` (GQA) is the right counter: on Llama-3.3-70B it is 8
    against 64 attention heads, i.e. 8× less cache than the model size would
    suggest. The KV cache stays in bf16 even when the weights are in 4-bit:
    quantisation does not shrink it."""
    seq_len = seq_len or MAX_SEQ_LENGTH
    cfg = model.config
    cfg = getattr(cfg, "text_config", cfg)      # multimodal models (Gemma, MedGemma)
    layers = getattr(cfg, "num_hidden_layers", 0)
    n_heads = getattr(cfg, "num_attention_heads", 0)
    kv_heads = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or (
        getattr(cfg, "hidden_size", 0) // n_heads if n_heads else 0
    )
    if not (layers and kv_heads and head_dim):
        return float("nan")
    return 2 * layers * kv_heads * head_dim * seq_len * batch_size * 2 / 1e9


def vram_report(label: str = "") -> dict:
    """Break the VRAM footprint down: weights (measured) + KV cache (estimated)."""
    weights = torch.cuda.memory_allocated() / 1e9
    kv = estimate_kv_cache_gb()
    total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = VRAM_BUDGET_GB or total_gpu

    print(f"  VRAM {label}: weights {weights:.1f} GB + KV cache ~{kv:.1f} GB"
          f" (at {MAX_SEQ_LENGTH} tokens) = ~{weights + kv:.1f} GB / {budget:.0f} GB")
    if weights + kv > budget:
        overshoot = weights + kv - budget
        # affordable seq_len = budget left after the weights, converted to tokens
        kv_per_token = kv / MAX_SEQ_LENGTH if MAX_SEQ_LENGTH else 0
        affordable = int((budget - weights) / kv_per_token) if kv_per_token > 0 else 0
        print(f"  ⛔ exceeds the budget by {overshoot:.1f} GB"
              f" — reduce MAX_SEQ_LENGTH to ~{affordable} tokens, or change model")
    return {"weights_gb": round(weights, 1), "kv_cache_gb": round(kv, 1),
            "budget_gb": budget, "fits": bool(weights + kv <= budget)}


def load_model(model_name: str) -> None:
    global model, tokenizer
    free_memory()
    report_memory("before ")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    print(f"  loaded in {time.time()-t0:.0f}s")
    report_memory("after  ")
    vram_report("estimated")

def _tok():
    """The real text tokenizer. Multimodal models (medgemma) return a Processor
    that wraps the tokenizer in .tokenizer; the others already are tokenizers
    (getattr then falls back to the object itself)."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def _to_inputs(out):
    """Normalise apply_chat_template's output: Processors may return a formatted
    STRING instead of tensors -> we retokenise it ourselves."""
    if isinstance(out, str):
        return _tok()(out, return_tensors="pt", add_special_tokens=False).to(model.device)
    return out.to(model.device)


def count_template_tokens(msgs):
    """TOKEN count of the chat template, robust to the 3 output shapes: list of
    ids, nested list [[ids]], or string (Processor)."""
    out = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    if isinstance(out, str):
        out = _tok()(out)["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return len(out)


def _apply_template(messages):
    """Fallback 1: templates without `enable_thinking` -> retry without it."""
    try:
        return _to_inputs(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=True,
        ))
    except (TypeError, ValueError, KeyError) as e:
        return _to_inputs(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ))


def _build_inputs(full_text: str):
    """Fallback 2: template with no 'system' role -> merged into the 'user' turn."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(full_text)},
    ]
    try:
        return _apply_template(messages)
    except Exception as e:
        if "system" in str(e).lower():
            merged = [{
                "role": "user",
                "content": SYSTEM_PROMPT + "\n\n" + build_user_prompt(full_text),
            }]
            return _apply_template(merged)
        raise

import json
import re


def call_llm_extraction(full_text: str, verbose: bool = False) -> dict:
    fixed_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt("")},
    ]
    fixed_tokens = count_template_tokens(fixed_messages)                      # ① Processor-safe
    raw_tokens = compute_number_of_tokens(full_text, _tok())                  # ② real tokenizer
    if verbose:
        print(f"  fixed prompt: {fixed_tokens / MAX_SEQ_LENGTH:.0%}  |  raw note: {raw_tokens / MAX_SEQ_LENGTH:.0%}  (window {MAX_SEQ_LENGTH} tokens)")

    note_budget = MAX_SEQ_LENGTH - fixed_tokens - MAX_NEW_TOKENS              # ③ inline
    full_text = truncate_text_to_token_budget(full_text, note_budget, _tok()) # ② real tokenizer
    n_truncated = max(0, raw_tokens - note_budget)
    if n_truncated > 0 and verbose:
        print(f"  ⚠️ note truncated: {n_truncated} tokens dropped ({n_truncated/raw_tokens:.0%})")

    fail = {"CVC": None, "evidence": None, "uncertain": None,
            "n_truncated": n_truncated, "raw": ""}

    try:
        inputs = _build_inputs(full_text)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                                              # greedy: deterministic
                pad_token_id=_tok().eos_token_id,                             # ④ a Processor has no eos_token_id
            )
        raw = _tok().decode(                                                  # ④ same for decode
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {**fail, "error": "oom", "raw": str(e)[:300]}
    except Exception as e:
        return {**fail, "error": type(e).__name__, "raw": str(e)[:300]}

    # "Thinking" models (Qwen3) emit a <think>…</think> block before the answer;
    # if they exhaust MAX_NEW_TOKENS inside it, the tag stays open — hence the
    # second re.sub, which cuts an unclosed <think>.
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    clean = re.sub(r"<think>.*$", "", clean, flags=re.DOTALL).strip()

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return {**fail, "error": "no_json", "raw": raw[-300:]}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {**fail, "error": "bad_json", "raw": match.group()[:300]}

    parsed.setdefault("error", "")
    parsed["n_truncated"] = n_truncated
    parsed["raw"] = ""
    return parsed

VERIFY_PATH = RESULTS_DIR / "verify.csv"

# 6 columns, in the order they are read: which problem, for which model, on which
# note, what the model produced, and what the note actually says.
VERIFY_COLS = ["verdict", "model", "id", "mismatch", "model_output", "note_context"]

# Device keywords: used to centre the note excerpt shown in verify.csv on the
# area that talks about catheters, rather than on the note header.
_DEVICE_HINT = re.compile(
    r"(picc|central line|central venous|cvc|cvl|\bij\b|port-?a-?cath|mediport|"
    r"hickman|broviac|groshong|vas-?cath|quinton|permcath|shaldon|swan-?ganz|"
    r"cordis|introducer|dialysis catheter|hd line|catheter)",
    re.IGNORECASE,
)


def _note_context(source: str, evidence: str, width: int = 200) -> str:
    """Note excerpt to show next to the quotation.

    Priority: the area that actually contains the quotation; failing that the
    first mention of a device; failing that the start of the note."""
    flat = " ".join(str(source).split())
    if not flat:
        return ""
    ev = " ".join(str(evidence).split())
    pos = flat.lower().find(ev.lower()[:60]) if len(ev) >= 15 else -1
    if pos < 0:
        # Quotation not found (the hallucination case): show the area of the note
        # that talks about a device, the only one useful to decide. The
        # administrative header of the first 400 characters teaches nothing.
        m = _DEVICE_HINT.search(flat)
        if m:
            pos = m.start()
        else:
            return "(no device mentioned in the note)"
    start = max(0, pos - width // 3)
    return flat[start:start + width]


def _verdict(row) -> str:
    if row["error"]:
        return "parse_failed"
    if not row["evidence"]:
        return "no_evidence"
    if not row["evidence_grounded"]:
        return "hallucination"
    if row["gt_catheter"] is not None and bool(row["gt_catheter"]) != bool(row["pred_catheter"]):
        return "wrong_class"
    return "ok"


# Severity order: the most serious is read first.
_SEVERITY = {"hallucination": 0, "parse_failed": 1, "no_evidence": 2, "wrong_class": 3}


def save_verify(label: str, results_df: pd.DataFrame) -> pd.DataFrame:
    """Stack the problem rows into verify.csv.

    The file is meant to be read as it is in a spreadsheet: only the cases that
    go wrong, 6 columns, sorted from most to least serious. The full count
    (including the `ok`) stays in `comparison.csv`."""
    rows = results_df.copy()
    rows.insert(0, "model_id", label)
    rows["verdict"] = rows.apply(_verdict, axis=1)
    rows = rows[rows.verdict != "ok"].copy()          # ← problems only
    if not len(rows):
        rows = pd.DataFrame(columns=VERIFY_COLS)
    else:
        src_by_id = dict(zip(df["id_note"], df["input"]))

        # A single "expected vs obtained" column rather than two booleans to cross
        rows["mismatch"] = [
            "-" if bool(g) == bool(pr) else ("missed a CVC" if g else "invented a CVC")
            for g, pr in zip(rows.gt_catheter, rows.pred_catheter)
        ]
        # A single text column: whatever the model produced, whatever its type
        rows["model_output"] = [
            (str(r.raw)[:200] if r.error else (r.evidence[:200] or "(empty)"))
            for r in rows.itertuples()
        ]
        rows["note_context"] = [
            _note_context(str(src_by_id.get(i, "")), e)
            for i, e in zip(rows["id"], rows["evidence"])
        ]
        rows["model"] = rows.model_id.str.split("/").str[-1]
        rows = rows[VERIFY_COLS]

    if VERIFY_PATH.exists():
        prev = pd.read_csv(VERIFY_PATH)
        prev = prev[prev["model"] != label.split("/")[-1]]    # replaces the re-run model
        rows = pd.concat([prev, rows], ignore_index=True)

    rows = rows.sort_values(
        ["verdict", "model"], key=lambda s: s.map(_SEVERITY).fillna(9) if s.name == "verdict" else s
    )
    rows.to_csv(VERIFY_PATH, index=False)
    return rows


def verify_summary() -> pd.DataFrame:
    """Model × verdict table. verify.csv only holds the problems: the `ok` are
    derived from the note total read in comparison.csv."""
    comparison_path = RESULTS_DIR / "comparison.csv"
    if not comparison_path.exists():
        print("no comparison.csv yet")
        return pd.DataFrame()
    totals = pd.read_csv(comparison_path)
    totals["model"] = totals.model.str.split("/").str[-1]
    totals = totals.set_index("model")["n"]

    problems = pd.read_csv(VERIFY_PATH) if VERIFY_PATH.exists() else pd.DataFrame()
    if len(problems):
        pivot = problems.pivot_table(index="model", columns="verdict", values="id",
                                     aggfunc="count", fill_value=0)
    else:
        pivot = pd.DataFrame(index=totals.index)
    pivot = pivot.reindex(totals.index, fill_value=0)
    for c in ["hallucination", "parse_failed", "no_evidence", "wrong_class"]:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[["hallucination", "parse_failed", "no_evidence", "wrong_class"]].fillna(0).astype(int)
    pivot.insert(0, "ok", totals - pivot.sum(axis=1))
    return pivot.sort_values("ok", ascending=False)

import time
from tqdm.auto import tqdm


def model_slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-")


def _normalise(s: str) -> str:
    """Loose note/quotation comparison: case, spaces and line breaks must not
    make an exact quotation look like a hallucination."""
    return " ".join(str(s).lower().split())


# Wordings the prompt explicitly ALLOWS when nothing is documented: these are
# not quotations, so looking for them in the note makes no sense. Without this
# exemption, any model that follows the instruction to the letter would be
# counted as hallucinating on every negative case.
_ABSENCE_PHRASES = (
    "no central venous device documented",
    "no central venous device",
    "no central line",
    "none documented",
    "not documented",
    "no evidence",
    "n/a",
    "none",
)


def evidence_is_grounded(evidence: str, source: str) -> bool:
    """Is the quotation really in the note?

    The model often quotes several passages separated by '...' or line breaks:
    we require at least one substantial fragment (>= 15 characters) to be found
    verbatim in the source. The absence wordings prescribed by the prompt are
    exempt."""
    if not evidence:
        return False
    ev = _normalise(evidence)
    if any(ev.startswith(p) or ev == p for p in _ABSENCE_PHRASES):
        return True
    src = _normalise(source)
    if ev in src:                                   # quotation in one piece
        return True
    # Composite quotations: models readily concatenate several authentic passages
    # from the note, separated by '...', ';', a line break or a sentence end.
    # Each piece has to be judged separately, otherwise a montage of exact
    # quotations would pass for an invention.
    fragments = [f for f in re.split(r"\.\.\.|[;\n]|\s\|\s|(?<=\.)\s+", evidence)
                 if len(f.strip()) >= 15]
    if not fragments:
        return ev in src
    grounded = [_normalise(f) in src for f in fragments]
    # at least half the fragments have to be found: tolerates a rephrased linking
    # sentence, but not a mostly invented quotation
    return sum(grounded) >= max(1, len(grounded) // 2)


def run_and_evaluate(model_name: str, save_as: str = None) -> pd.DataFrame:
    load_model(model_name)                    # ALWAYS loads with the real HF name
    label = save_as or model_name             # label for CSV + comparison (distinct if given)

    records = []
    for i, row in enumerate(tqdm(df.to_dict("records"), desc=label)):
        start = time.time()
        pred = call_llm_extraction(row["input"], verbose=(i == 0))   # token detail on the 1st note
        duration = time.time() - start
        evidence = (pred.get("evidence") or "").strip()
        records.append({
            "id": row["id_note"],
            "gt_catheter": (row["result"] == "Y") if has_ground_truth else None,
            "pred_catheter": pred.get("CVC") == "Yes",
            "uncertain": pred.get("uncertain"),
            "evidence": evidence[:300],
            "evidence_grounded": evidence_is_grounded(evidence, row["input"]),
            "error": pred.get("error", ""),
            "raw": pred.get("raw", ""),                              # raw output on failure
            "n_truncated": pred.get("n_truncated", 0),
            "duration_s": round(duration, 1),
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_DIR / f"{model_slug(label)}.csv", index=False)     # ← label

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    vram = vram_report("measured")
    ok = results_df.error.fillna("") == ""                         # rows actually parsed
    summary = {"model": label, "n": len(results_df)}               # ← label
    if has_ground_truth:
        gt, pr = results_df.gt_catheter.astype(bool), results_df.pred_catheter.astype(bool)
        tp = int((gt & pr).sum())
        fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum())
        tn = int((~gt & ~pr).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary.update(tp=tp, fp=fp, fn=fn, tn=tn,
                       precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3),
                       accuracy=round((tp + tn) / len(results_df), 3))

    # Format quality, independent of how correct the classification is
    summary["parse_failed"] = int((~ok).sum())
    summary["no_evidence"] = int((ok & (results_df.evidence == "")).sum())
    summary["hallucinated_evidence"] = int(
        (ok & (results_df.evidence != "") & ~results_df.evidence_grounded).sum()
    )
    summary["truncated_notes"] = int((results_df.n_truncated > 0).sum())
    summary["avg_duration_s"] = round(results_df.duration_s.mean(), 1)
    summary["total_min"] = round(results_df.duration_s.sum() / 60, 1)
    summary["weights_gb"] = vram["weights_gb"]
    summary["kv_cache_gb"] = vram["kv_cache_gb"]
    summary["peak_vram_gb"] = round(peak_vram, 1)
    summary["vram_budget_gb"] = vram["budget_gb"]
    # The real peak includes weights + KV cache + activations: that is what
    # decides whether the model fits on the target, not the pre-run estimate.
    summary["fits_budget"] = bool(peak_vram <= vram["budget_gb"])

    # Immediate diagnosis: without it, a model that fails 100% of the time looks
    # like a model that always answers "No" (pred_catheter=False in both cases).
    if summary["parse_failed"]:
        kinds = results_df.loc[~ok, "error"].value_counts().to_dict()
        print(f"  ⚠️ {summary['parse_failed']}/{len(results_df)} parsing failures: {kinds}")
    if summary["no_evidence"]:
        print(f"  ⚠️ {summary['no_evidence']}/{len(results_df)} answers with no quotation (valid JSON but empty evidence)")
    if summary["hallucinated_evidence"]:
        print(f"  ⚠️ {summary['hallucinated_evidence']}/{len(results_df)} quotations not found in the source note")
    if not summary["fits_budget"]:
        print(f"  ⛔ peak VRAM {summary['peak_vram_gb']} GB > budget {summary['vram_budget_gb']:.0f} GB"
              f" — this model DOES NOT FIT on the target")
    if has_ground_truth:
        print(f"  F1={summary['f1']}  P={summary['precision']}  R={summary['recall']}"
              f"  | peak VRAM {summary['peak_vram_gb']} GB  | {summary['total_min']} min")

    comparison_path = RESULTS_DIR / "comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    if not comparison.empty:
        comparison = comparison[comparison["model"] != label]                   # ← label
    comparison = pd.concat([comparison, pd.DataFrame([summary])], ignore_index=True)
    comparison.to_csv(comparison_path, index=False)

    save_verify(label, results_df)          # single file, all notes / all models
    return comparison

def main() -> None:
    global RESULTS_DIR, VERIFY_PATH, MAX_SEQ_LENGTH, df, has_ground_truth

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, type=Path,
                    help="CSV with columns id_note, input, and result (Y/N) if evaluating")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", default=None, help="label used in comparison.csv")
    ap.add_argument("--max-seq-length", type=int, default=16384)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--gold", type=Path, default=None,
                    help="CSV of reference labels (id_note, result), joined to the corpus "
                         "when the corpus carries none — so an F1 can be measured")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU detected: 4-bit inference requires one")
    print(torch.cuda.get_device_name(0),
          "- capability", torch.cuda.get_device_capability(0))

    df = pd.read_csv(args.data)
    assert {"id_note", "input"}.issubset(df.columns), \
        "expected columns: id_note, input (+ result in order to evaluate)"

    # With no labels the extraction still runs: it produces the predictions, and
    # verify.csv still detects hallucinations and format failures, which require no
    # ground truth. Only the metrics (F1, precision, recall) need a `result` column.
    if args.gold is not None and "result" not in df.columns:
        labels = pd.read_csv(args.gold)[["id_note", "result"]]
        before = len(df)
        df = df.merge(labels, on="id_note")
        if len(df) < before:
            print(f"⚠️  {before - len(df)} note(s) with no label in "
                  f"{args.gold.name}, dropped")

    has_ground_truth = "result" in df.columns
    print(f"{args.data.name}: {df.shape[0]} notes, ground truth {has_ground_truth}")

    RESULTS_DIR = args.results_dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_PATH = RESULTS_DIR / "verify.csv"
    MAX_SEQ_LENGTH = args.max_seq_length
    print(f"MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}  |  results -> {RESULTS_DIR}")

    run_and_evaluate(args.model, save_as=args.label)


if __name__ == "__main__":
    main()
