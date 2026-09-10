"""Reconcile an Anguli output tree after an interrupted run.

Anguli writes each finger's clean master, its N impressions and its metadata into
separate parallel subtrees, so a run killed mid-finger leaves a torn record: the
master exists but one or more impressions do not. Training on that silently pairs
a clean image with a missing counterpart, or drops fingers unevenly across
impressions -- neither fails loudly.

This finds the fingers present in *every* subtree, reports the torn ones, and
optionally prunes them. It also writes the `generation.json` provenance file that
`generate_anguli.py` never got to write, marked as a partial run, so the corpus
still carries its own history.

    python scripts/reconcile_anguli.py data/processed/anguli_20k
    python scripts/reconcile_anguli.py data/processed/anguli_20k --prune
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF_INT = re.compile(r'<(\w+)\s+value="([^"]*)"')


def finger_ids(subtree: Path, suffix: str) -> set[tuple[str, str]]:
    """Identity of a finger is (batch directory, file stem), e.g. ('fp_6', '585')."""
    return {
        (p.parent.name, p.stem)
        for p in subtree.rglob(f"*{suffix}")
        if p.is_file()
    }


def parse_conf(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for key, value in CONF_INT.findall(path.read_text(errors="replace")):
        out[key] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="an Anguli output directory")
    ap.add_argument("--prune", action="store_true",
                    help="delete files belonging to incomplete fingers")
    ap.add_argument("--itype", default="png")
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    if not (corpus / "Fingerprints").is_dir():
        print(f"not an Anguli output tree: {corpus}")
        return 1

    image_trees = [corpus / "Fingerprints"] + sorted(
        d for d in corpus.glob("Impression_*") if d.is_dir()
    )
    meta_tree = corpus / "Meta Info"

    sets: dict[str, set[tuple[str, str]]] = {
        t.name: finger_ids(t, f".{args.itype}") for t in image_trees
    }
    if meta_tree.is_dir():
        sets["Meta Info"] = finger_ids(meta_tree, ".txt")

    print(f"{corpus.name}:")
    for name, ids in sets.items():
        print(f"  {name:14} {len(ids):6}")

    complete = set.intersection(*sets.values())
    union = set.union(*sets.values())
    torn = union - complete
    print(f"\n  complete fingers: {len(complete)}")
    print(f"  torn fingers:     {len(torn)}")
    for fid in sorted(torn)[:10]:
        missing = [name for name, ids in sets.items() if fid not in ids]
        print(f"    {fid[0]}/{fid[1]}  missing from: {', '.join(missing)}")
    if len(torn) > 10:
        print(f"    ... and {len(torn) - 10} more")

    if torn and args.prune:
        removed = 0
        for batch, stem in sorted(torn):
            for tree in image_trees:
                p = tree / batch / f"{stem}.{args.itype}"
                if p.exists():
                    p.unlink()
                    removed += 1
            p = meta_tree / batch / f"{stem}.txt"
            if p.exists():
                p.unlink()
                removed += 1
        print(f"\n  pruned {removed} file(s) from {len(torn)} torn finger(s)")
    elif torn:
        print("\n  re-run with --prune to remove them")

    conf = parse_conf(corpus / "Anguli.conf")
    requested = int(conf.get("NumFingerprints", 0) or 0)
    n_impressions = len(image_trees) - 1
    provenance = {
        "corpus": corpus.name,
        "reconciled": date.today().isoformat(),
        "partial_run": bool(requested and len(complete) < requested),
        "fingers_requested": requested or None,
        "fingers_complete": len(complete),
        "impressions_per_finger": n_impressions,
        "images": len(complete) * (n_impressions + 1),
        "seed": conf.get("seedValue"),
        "image_type": conf.get("ImageType"),
        "torn_fingers_found": len(torn),
        "torn_fingers_pruned": len(torn) if args.prune else 0,
        "anguli_conf": conf,
        "note": (
            "Anguli-side generation only; artefacts are applied downstream by "
            "fpe.degradation. Interrupted run reconciled by scripts/reconcile_anguli.py."
        ),
    }
    out = corpus / "generation.json"
    out.write_text(json.dumps(provenance, indent=2))
    print(f"\n  wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print(f"  usable: {len(complete)} fingers, "
          f"{len(complete) * (n_impressions + 1)} images "
          f"({n_impressions} impressions each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
