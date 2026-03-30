from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.coin_pipeline import draw_many_and_process, get_image_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
GROUND_TRUTHS_PATH = DATASET_ROOT / "ground_truths.txt"

app = FastAPI(title="Coin Counter API - phase2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _norm(path_text: str) -> str:
    """Normalize path text to lowercase dataset-relative shape."""
    fixed = path_text.strip().replace("\\", "/").lower()
    i = fixed.find("dataset/")
    return fixed[i:] if i >= 0 else fixed


@lru_cache(maxsize=1)
def _load_gt() -> dict[str, int]:
    """Load GT labels from text file."""
    if not GROUND_TRUTHS_PATH.exists():
        return {}
    out: dict[str, int] = {}
    for raw in GROUND_TRUTHS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        p, c = line.rsplit(",", 1)
        try:
            out[_norm(p)] = int(c.strip())
        except ValueError:
            continue
    return out


def _attach_gt(item: dict, gt_map: dict[str, int]) -> dict:
    """Attach GT + error fields to one item."""
    copied = deepcopy(item)
    gt = gt_map.get(_norm(copied.get("image_path", "")))
    copied["ground_truth_count"] = gt
    if gt is None:
        copied["error"] = None
        copied["absolute_error"] = None
    else:
        err = int(copied["coin_count"]) - int(gt)
        copied["error"] = err
        copied["absolute_error"] = abs(err)
    return copied


@app.get("/api/health")
def health() -> dict[str, str]:
    """Simple health ping."""
    return {"status": "ok"}


@app.get("/api/splits")
def splits() -> dict[str, list[str]]:
    """List available dataset folders."""
    subdirs = sorted([p.name for p in DATASET_ROOT.iterdir() if p.is_dir()])
    return {"splits": ["all", *subdirs]}


@app.get("/api/draw")
def draw(split: str = Query(default="all"), seed: int | None = Query(default=None), count: int = Query(default=10, ge=1, le=25)) -> dict:
    """Draw random samples and attach GT error per image."""
    try:
        if split != "all":
            available = {p.parent.name.lower() for p in get_image_paths(DATASET_ROOT, "all")}
            if split.lower() not in available:
                raise HTTPException(status_code=404, detail=f"Split '{split}' not found in dataset.")
        gt_map = _load_gt()
        items = draw_many_and_process(dataset_root=DATASET_ROOT, split=split, seed=seed, count=count)
        items = [_attach_gt(item, gt_map) for item in items]
        return {"count": len(items), "items": items, "split": split, "seed": seed}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

