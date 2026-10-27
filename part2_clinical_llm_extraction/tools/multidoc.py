"""Split a note into several documents, and re-join them before extraction.

`merge_records` handles both cases without being told which: if a note spans
several rows it is rebuilt with `=== DOCUMENT i/n ===` headers; if it fits on one
row it comes back out as it was, with no header. An un-split corpus therefore
passes through the function unchanged.

Re-joining without headers (`delimiter=False`) gives back the original text
character for character.

    python3 multidoc.py CATH-LLM/litlle_sample.csv 2    # -> ..._split2.csv
"""

import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)   # a merged note exceeds the default limit

# Section header: alone on its line, nothing after the ':'. "Brief Hospital Course:"
# is one; "Name:  ___  Unit No: ___" is not, since its value sits on the same line —
# so the demographic preamble stays in one block instead of being chopped up.
SECTION = re.compile(r"^[ \t]*[A-Z][^\n:]{2,60}:[ \t]*\r?$")


def split_text(text: str, n_parts: int = 2) -> list[str]:
    """Cut on the section boundaries closest to the length quantiles."""
    blocks, current = [], []
    for line in text.splitlines(keepends=True):
        if current and SECTION.match(line):
            blocks.append("".join(current))
            current = [line]
        else:
            current.append(line)
    blocks.append("".join(current))

    if len(blocks) < n_parts:                    # note without enough sections
        step = max(1, -(-len(text) // n_parts))
        blocks = [text[i:i + step] for i in range(0, len(text), step)]
    if len(blocks) < n_parts:                    # empty or tiny note
        return blocks or [text]

    sizes = [0]
    for block in blocks:
        sizes.append(sizes[-1] + len(block))

    cuts, start = [], 0
    for k in range(1, n_parts):
        target = sizes[-1] * k / n_parts
        # bounding the candidates leaves at least one block for each remaining part
        window = range(start + 1, len(blocks) - (n_parts - k) + 1)
        start = min(window, key=lambda i: abs(sizes[i] - target))
        cuts.append(start)

    bounds = [0, *cuts, len(blocks)]
    return ["".join(blocks[a:b]) for a, b in zip(bounds, bounds[1:])]


def split_records(records: list[dict], n_parts: int = 2) -> list[dict]:
    """One row per note -> one row per document, other columns copied through."""
    out = []
    for record in records:
        parts = split_text(str(record["input"]), n_parts)
        for index, part in enumerate(parts, start=1):
            out.append({**{k: v for k, v in record.items() if k != "input"},
                        "doc_index": index, "n_docs": len(parts), "input": part})
    return out


def merge_records(records: list[dict], delimiter: bool = True) -> list[dict]:
    """One row per note, whether the notes are split or not.

    The documents are put back in `doc_index` order, not in file order.
    """
    groups: dict = {}
    for record in records:
        groups.setdefault(record["id_note"], []).append(record)

    out = []
    for docs in groups.values():
        docs.sort(key=lambda r: int(r.get("doc_index", 1)))
        texts = [str(d["input"]) for d in docs]
        row = {k: v for k, v in docs[0].items() if k not in ("input", "doc_index", "doc_id")}
        row["n_docs"] = len(texts)
        row["input"] = "".join(texts) if len(texts) == 1 or not delimiter else "\n\n".join(
            f"=== DOCUMENT {i}/{len(texts)} ===\n{t.strip()}" for i, t in enumerate(texts, 1))
        out.append(row)
    return out


if __name__ == "__main__":
    source = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with source.open(newline="", encoding="utf-8") as handle:
        rows = split_records(list(csv.DictReader(handle)), n)
    out = source.with_name(f"{source.stem}_split{n}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} documents -> {out}")
