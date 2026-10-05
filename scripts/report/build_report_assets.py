"""Build the image panels and metric charts for the stage-wise progress report.

Every enhanced image is produced live from the stage's committed checkpoint, every
aggregate number is read from results/registry/runs.jsonl (PROJECT_RULES.md rule 2), and the
hero print's per-image NFIQ 2 is computed with the same wrapper the benchmark uses.

    .venv/Scripts/python.exe scripts/report/build_report_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from fpe.data.convert import load_greyscale  # noqa: E402
from fpe.models.wafen import Wafen, WafenConfig  # noqa: E402

torch.set_num_threads(4)

OUT = ROOT / "results" / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)

HERO = ROOT / "data" / "raw" / "fvc2004" / "db1_b" / "101_1.tif"
ZOOM = (240, 240, 368, 368)  # x0, y0, x1, y1 -- the faint, fragmented lower region

STAGES = [
    ("raw", None, "Raw degraded input"),
    ("s1", "data/work/wafen/best.pt", "Stage 1: corrected synthesis"),
    ("s2", "data/work/wafen_ft/best.pt", "Stage 2: + teacher distillation"),
    ("s3", "data/work/wafen_mn/best.pt", "Stage 3: + minutia-aware loss"),
    ("s4", "data/work/wafen_pc/best.pt", "Stage 4: + pair consistency"),
    ("s5", "data/work/wafen_da/best.pt", "Stage 5: domain alignment"),
]

ACCENT = (217, 89, 38)  # palette accent (orange) for boxes and arrows
INK = (11, 11, 11)


def enhance(checkpoint: Path, img01: np.ndarray) -> np.ndarray:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Wafen(WafenConfig(**state["config"])).eval()
    model.load_state_dict(state["model"])
    stride = 2 ** (model.config.depth - 1)
    h, w = img01.shape
    padded = np.pad(img01, ((0, -h % stride), (0, -w % stride)), constant_values=1.0)
    with torch.no_grad():
        out = model(torch.from_numpy(padded)[None, None].float())
    ridge = out.ridge[0, 0, :h, :w].numpy()
    mask = out.segmentation[0, 0, :h, :w].numpy() >= 0.5
    ink = np.where(mask, 1.0 - ridge, 1.0)
    return np.round(ink * 255).astype(np.uint8)


def panel(image: np.ndarray, name: str) -> Path:
    """Main view with the zoom region boxed, plus a 3x inset joined by an arrow."""
    x0, y0, x1, y1 = ZOOM
    main = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(main)
    draw.rectangle(ZOOM, outline=ACCENT, width=3)

    crop = Image.fromarray(image[y0:y1, x0:x1]).convert("RGB")
    scale = 3
    inset = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
    d2 = ImageDraw.Draw(inset)
    d2.rectangle([0, 0, inset.width - 1, inset.height - 1], outline=ACCENT, width=5)

    gap = 48
    canvas = Image.new("RGB", (main.width + gap + inset.width,
                               max(main.height, inset.height)), (252, 252, 251))
    canvas.paste(main, (0, (canvas.height - main.height) // 2))
    ix = main.width + gap
    iy = (canvas.height - inset.height) // 2
    canvas.paste(inset, (ix, iy))

    d3 = ImageDraw.Draw(canvas)  # arrow: zoom box edge -> inset edge
    ay = (canvas.height - main.height) // 2 + (y0 + y1) // 2
    by = iy + inset.height // 2
    d3.line([(x1 + 4, ay), (ix - 6, by)], fill=ACCENT, width=5)
    d3.polygon([(ix - 4, by), (ix - 26, by - 12), (ix - 26, by + 12)], fill=ACCENT)

    path = OUT / f"panel_{name}.png"
    canvas.save(path)
    return path


def registry_numbers() -> dict:
    """Latest aggregate fvc2004 row per stage/baseline, straight from the registry."""
    want = {"none": "raw", "GBFEN": "gbfen", "SNFEN": "snfen"}
    ckpt_want = {"wafen": "s1", "wafen_ft": "s2", "wafen_mn": "s3",
                 "wafen_pc": "s4", "wafen_da": "s5"}
    out: dict = {}
    for line in (ROOT / "results" / "registry" / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("experiment") != "baseline-nbis" or r.get("dataset") != "fvc2004":
            continue
        cfg = r.get("config") or {}
        key = None
        if r["method"] in want:
            key = want[r["method"]]
        elif r["method"] == "WAFEN" and not cfg.get("blend") \
                and cfg.get("abstain_threshold") is None:
            key = ckpt_want.get(Path(cfg.get("checkpoint", "x/y")).parent.name)
        if key:
            out[key] = r["metrics"]  # later rows supersede
    return out


def main() -> int:
    img = load_greyscale(HERO)
    img01 = img.astype(np.float32) / 255.0

    stage_files = {}
    for name, ckpt, _ in STAGES:
        arr = img if ckpt is None else enhance(ROOT / ckpt, img01)
        raw_path = OUT / f"img_{name}.png"
        Image.fromarray(arr).save(raw_path)
        panel(arr, name)
        stage_files[name] = raw_path
        print(f"  {name}: panel written")

    from fpe.metrics.nfiq2 import score_images
    scores = score_images([stage_files[n] for n, _, _ in STAGES])
    hero_nfiq = {name: (int(s.score) if s.ok else None)
                 for (name, _, _), s in zip(STAGES, scores)}

    data = {"hero": str(HERO.relative_to(ROOT)), "zoom": ZOOM,
            "hero_nfiq2": hero_nfiq, "registry": registry_numbers()}
    (OUT / "report_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(hero_nfiq))
    print("registry keys:", sorted(data["registry"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
