"""Translate a corpus of notes, one paragraph per call. No format assumptions.

Written to be reusable as-is on any text corpus, by anyone: the only thing this
script assumes about a note is that it contains blank lines. No header table, no
section pattern, no laboratory-line detector — nothing that comes from MIMIC.

    .venv312/bin/python tools/traduire_paragraphes.py \\
        --data results/notes_croate_milmmt-12b.csv \\
        --src Croatian --tgt English \\
        --out results/notes_retour_anglais.csv

Why the paragraph is the right unit, measured on the 249 notes of
CATH-LLM/exampleNotes_250.xlsx and on the Croatian this script produced from them
(results/notes_croate_paragraphes.csv):

    paragraphs per note         38 (median), 121 at most
    size of a paragraph         109 chars in English, 133 in Croatian
    above 1000 chars            3.8 % in English, 4.9 % in Croatian

And above all: one paragraph in equals one paragraph out. The paragraph survives
translation, so it is a stable unit — which is true of neither the sentence nor the
line. Note that this script's own output has *more* paragraphs than its input
(10 471 against 10 031 in English), because a paragraph above the character ceiling
is cut into sentences and each piece is re-joined as a paragraph of its own.

The trap this splitting avoids, and which cost a whole run: filling each call up to
the model's context window. A translation model stops when it judges it has produced
a complete translation, by emitting its end token — and it then **ignores the rest of
the input, raising no error**. The phenomenon has a name in the literature, *length
bias*: quality collapses as soon as you decode much longer than what was seen in
training. What bounds the useful input is therefore not the context window (131,072
tokens here) but the training length (sentence pairs). Measured on this corpus:

    calls of ~100 chars    loss ~3.5 %
    calls of 3000 chars    14 % of notes cut short, 13 % of redactions lost
    whole note (10,000)    ~40 % of every note lost, systematically
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent

# De-identification redactions. Specific to MIMIC, hence switchable: this is the only
# concession to the corpus format, and it has no effect on a text that contains none.
MARK = 77700
REDACTION = re.compile(r"_{3,}")


def paragraphs(note: str) -> list[str]:
    """Note -> paragraphs, one per blank line. No other assumption.

    Blank lines are preserved implicitly: re-joining puts them back, so the structure
    of the document survives without anyone having to describe it."""
    out, current = [], []
    for line in str(note).splitlines():
        if line.strip():
            current.append(line)
        elif current:
            out.append("\n".join(current))
            current = []
    if current:
        out.append("\n".join(current))
    return out


def sentences(text: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph, never cutting in the middle of a sentence.

    Only fires on paragraphs above the ceiling: at the applied ceiling of 1000
    characters, 377 paragraphs out of 10 031 in English (3.8 %) and 516 out of
    10 483 in Croatian (4.9 %). A sentence cut in half yields a translation that is
    wrong, not merely incomplete, so the sentence boundary is an absolute minimum."""
    if len(text) <= max_chars:
        return [text]
    pieces, buffer = [], ""
    for p in re.split(r"(?<=[.!?])\s+", text):
        if buffer and len(buffer) + len(p) + 1 > max_chars:
            pieces.append(buffer)
            buffer = p
        else:
            buffer = f"{buffer} {p}".strip() if buffer else p
    if buffer:
        pieces.append(buffer)
    return pieces


def split(note: str, max_chars: int = 1000) -> list[str]:
    """One call per paragraph; paragraphs above the ceiling are cut into sentences."""
    return [m for p in paragraphs(note) for m in sentences(p, max_chars)]


def protect(text: str) -> tuple[str, int]:
    """`___` -> unique integers. A translation model copies numbers through."""
    n = 0

    def sub(_):
        nonlocal n
        n += 1
        return str(MARK + n)

    return REDACTION.sub(sub, text), n


def restore(text: str, n: int) -> tuple[str, int]:
    """The text with the `___` put back, and the number of markers lost.

    Forgiving: we put back what comes back and count the losses, rather than
    rejecting. Wholesale rejection only makes sense on a unit small enough to be
    replaced by its source without damage."""
    lost = 0
    for i in range(1, n + 1):
        m = str(MARK + i)
        if m in text:
            text = text.replace(m, "___", 1)
        else:
            lost += 1
    return re.sub(rf"{MARK // 100}\d\d", "___", text), lost


def budget(batch: list[str], ceiling: int = 2048) -> int:
    """Output tokens allowed for this batch.

    Computed per batch rather than fixed once and for all: a value set too low
    truncates **without an error**. The 2x margin over an estimate of 3.5 chars per
    token covers languages that tokenise badly, Croatian consuming ~15 % more tokens
    than English for the same content."""
    return min(ceiling, int(max(len(t) for t in batch) / 3.5 * 2.0) + 64)


def plan_vram(batch: int, max_chars: int, four_bit: bool, budget_gb: float) -> None:
    """Expected footprint, before loading. The target is an A100 40 GB.

    The KV cache is the item people forget: it stays in bf16 even with quantised
    weights, and it grows with the batch AND with the sequence length."""
    weights = 7.0 if four_bit else 24.0             # MiLMMT-46-12B, measured
    kv_per_token = 2 * 48 * 8 * 256 * 2 / 1e9       # layers, KV heads, dim, K+V, bf16
    tokens = int(max_chars / 3.5) + budget([" " * max_chars])
    kv = kv_per_token * tokens * batch
    total = (weights + kv) * 1.15                   # activation margin
    print(f"expected VRAM: weights {weights:.0f} GB + KV cache {kv:.1f} GB "
          f"({batch} x {tokens} tokens) = ~{total:.0f} GB / {budget_gb:.0f} GB target")
    if total > budget_gb:
        over = int(batch * budget_gb / total)
        print(f"  WARNING: exceeds the target. Reduce --batch to ~{over} "
              f"or pass --four-bit")


def load(model: str, four_bit: bool, batch: int, src: str, tgt: str):
    """Load the model once, and return the only function the rest of the code sees.

    Everything model-specific stays shut inside this closure — tokenizer, quantisation,
    prompt shape, batching. The caller receives `translate(texts) -> texts` and nothing
    else, which is what makes a translator swappable without touching the pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"        # on the right, the model continues the padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    mdl = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()
    print(f"loaded: {model} ({'4-bit' if four_bit else 'bf16'}, batch {batch}, "
          f"{src} -> {tgt})", flush=True)

    def prompt(t: str) -> str:
        """The prompt shape MiLMMT expects: a plain completion, no chat template.

        The language labels have to be the ones in the model's documentation
        (`English`, not `en`): an unrecognised label degrades the translation
        without raising anything."""
        return f"Translate this from {src} to {tgt}:\n{src}: {t}\n{tgt}:"

    def _batch(texts: list[str]) -> list[str]:
        """One generation call for one batch, and nothing else.

        No error handling here on purpose: an out-of-memory error has to reach `run`,
        which is the only place that knows how to split the batch and retry."""
        enc = tok([prompt(t) for t in texts], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(mdl.device)
        with torch.inference_mode():
            gen = mdl.generate(**enc, max_new_tokens=budget(texts),
                               do_sample=False, pad_token_id=tok.pad_token_id)
        # left padding: every sequence has the same input length, so a single slice
        # is enough to strip the prompt off the output.
        return [s.strip() for s in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                    skip_special_tokens=True)]

    def translate(texts: list[str]) -> list[str]:
        """Batches grouped by length, re-split on an out-of-memory error.

        Sorting by length is what makes the batch safe: without it, one long
        paragraph forces all its neighbours up to its size and the KV cache
        overflows. Recursive re-splitting bounds the damage of an overflow to the
        note being processed."""
        if not texts:
            return []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        output: dict[int, str] = {}

        def run(indices: list[int]) -> None:
            """Translate these indices, halving the batch on every out-of-memory error.

            Writes into `output` instead of returning, so the recursion has nothing to
            reassemble. A single index that still fails re-raises: it never will pass."""
            try:
                for txt, idx in zip(_batch([texts[i] for i in indices]), indices):
                    output[idx] = txt
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(indices) == 1:
                    raise
                m = len(indices) // 2
                print(f"    OOM on {len(indices)} -> re-splitting", flush=True)
                run(indices[:m])
                run(indices[m:])

        for d in range(0, len(order), batch):
            run(order[d:d + batch])
        return [output[i] for i in range(len(texts))]

    return translate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="CSV with columns id_note, input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="Croatian", help="source language, as the model names it")
    ap.add_argument("--tgt", default="English", help="target language")
    ap.add_argument("--model", default="xiaomi-research/MiLMMT-46-12B-v1.0")
    ap.add_argument("--max-chars", type=int, default=1000,
                    help="character ceiling per call (default 1000): 96 %% of "
                         "paragraphs fit under it, the rest are cut into sentences")
    ap.add_argument("--batch", type=int, default=24,
                    help="24 by default: the most that fits on an A100 40 GB in "
                         "bf16 with the activation margin")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--vram-gb", type=float, default=40.0,
                    help="VRAM of the TARGET machine (default 40, an A100)")
    ap.add_argument("--limit", type=int, default=None, help="process only N notes")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="one note in N, offset I — one process per GPU")
    ap.add_argument("--no-protect", action="store_true",
                    help="do not protect the `___` (corpus without redactions)")
    args = ap.parse_args()

    notes = pd.read_csv(ROOT / args.data)[["id_note", "input"]]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        notes = notes.iloc[i::n]
        print(f"shard {i}/{n}: {len(notes)} notes")

    # Check the splitting before spending any GPU: the redactions have to survive the
    # plain split/re-join round trip, with no translation involved.
    damaged = [r.id_note for r in notes.itertuples()
               if "\n\n".join(split(r.input, args.max_chars)).count("___")
               != str(r.input).count("___")]
    sizes = [len(m) for r in notes.itertuples() for m in split(r.input, args.max_chars)]
    print(f"splitting: {len(sizes)} calls, {len(sizes) / len(notes):.0f} per note, "
          f"median {sorted(sizes)[len(sizes) // 2]} chars, max {max(sizes)}")
    if damaged:
        print(f"  WARNING redactions altered by the splitting on {len(damaged)} notes")

    plan_vram(args.batch, args.max_chars, args.four_bit, args.vram_gb)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out).to_dict("records") if out.exists() else []
    already = {r["id_note"] for r in done}
    todo = notes[~notes.id_note.isin(already)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(already)} already done, {len(todo)} to go -> {out}", flush=True)
    if not len(todo):
        return

    translate = load(args.model, args.four_bit, args.batch, args.src, args.tgt)

    for k, row in enumerate(todo.itertuples(), 1):
        t0 = time.time()
        source = str(row.input)
        chunks = split(source, args.max_chars)

        if args.no_protect:
            protected, counts = list(chunks), [0] * len(chunks)
        else:
            protected, counts = map(list, zip(*(protect(m) for m in chunks)))

        rendered = translate(protected)

        parts, lost = [], 0
        for text, n in zip(rendered, counts):
            t, p = (text, 0) if not n else restore(text, n)
            parts.append(t)
            lost += p
        # Re-joining: one paragraph per line, separated by a blank line — the
        # structure of the original document, without having had to describe it.
        target = "\n\n".join(parts)

        done.append({
            "id_note": row.id_note, "input": target, "calls": len(chunks),
            "redactions_src": source.count("___"), "redactions_tgt": target.count("___"),
            "markers_lost": lost,
            "chars_src": len(source), "chars_tgt": len(target),
            "length_ratio": round(len(target) / max(1, len(source)), 3),
            "duration_s": round(time.time() - t0, 1),
        })
        pd.DataFrame(done).to_csv(out, index=False)
        d = done[-1]
        alert = ""
        if d["length_ratio"] < 0.7:
            alert = f"  WARNING ratio {d['length_ratio']}"
        elif lost:
            alert = f"  {lost} markers lost"
        print(f"  [{k}/{len(todo)}] {row.id_note} {d['duration_s']}s  {d['calls']} calls  "
              f"{d['chars_src']}->{d['chars_tgt']} chars (x{d['length_ratio']}){alert}",
              flush=True)

    res = pd.DataFrame(done)
    print(f"\n{len(res)} notes translated  {args.src} -> {args.tgt}")
    print(f"  length ratio  : median {res.length_ratio.median():.2f}, "
          f"min {res.length_ratio.min():.2f}")
    print(f"  notes under 0.7: {int((res.length_ratio < 0.7).sum())}")
    print(f"  markers lost  : {int(res.markers_lost.sum())} of "
          f"{int(res.redactions_src.sum())}")
    print(f"  duration: {res.duration_s.mean():.1f} s/note, {res.duration_s.sum() / 60:.0f} min")
    print("written:", out)


if __name__ == "__main__":
    main()
