from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.coin_pipeline import draw_many_and_process, estimate_coin_count, get_image_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
GROUND_TRUTHS_PATH = DATASET_ROOT / "ground_truths.txt"

app = FastAPI(title="Coin Counter API - phase3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _norm(path_text: str) -> str:
    """Normalize path text for matching GT keys."""
    fixed = path_text.strip().replace("\\", "/").lower()
    i = fixed.find("dataset/")
    return fixed[i:] if i >= 0 else fixed


@lru_cache(maxsize=1)
def _load_gt() -> dict[str, int]:
    """Load GT file once and cache it."""
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
    """Attach GT + error fields to one draw item."""
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


def _evaluate_split(split: str) -> dict:
    """Compute dataset-level metrics for one split."""
    img_paths = get_image_paths(dataset_root=DATASET_ROOT, split=split)
    gt_map = _load_gt()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(img_paths)))) as pool:
        future_to_path = {pool.submit(estimate_coin_count, path): path for path in img_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            pred = int(future.result())
            rel = str(path.relative_to(DATASET_ROOT.parent))
            gt = gt_map.get(_norm(rel))
            rows.append({"image_path": rel, "predicted_count": pred, "ground_truth_count": gt, "error": None if gt is None else pred - int(gt)})
    rows.sort(key=lambda x: x["image_path"])
    matched = [row for row in rows if row["ground_truth_count"] is not None]
    abs_errs = [abs(int(row["error"])) for row in matched]
    sq_errs = [int(row["error"]) ** 2 for row in matched]
    total_pred = int(sum(int(row["predicted_count"]) for row in matched))
    total_gt = int(sum(int(row["ground_truth_count"]) for row in matched))
    return {
        "split": split,
        "num_images": len(rows),
        "num_scored": len(matched),
        "mae": float(sum(abs_errs) / len(matched)),
        "rmse": float((sum(sq_errs) / len(matched)) ** 0.5),
        "total_error": total_pred - total_gt,
    }


@lru_cache(maxsize=12)
def _cached_eval(split: str) -> dict:
    """Memoized split evaluation."""
    return _evaluate_split(split)


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
    """Draw random samples and attach GT error."""
    gt_map = _load_gt()
    items = draw_many_and_process(dataset_root=DATASET_ROOT, split=split, seed=seed, count=count)
    items = [_attach_gt(item, gt_map) for item in items]
    return {"count": len(items), "items": items, "split": split, "seed": seed}


@app.get("/api/evaluation")
def evaluation(split: str = Query(default="all"), refresh: bool = Query(default=False)) -> dict:
    """Dataset metrics endpoint."""
    if refresh:
        _cached_eval.cache_clear()
    try:
        return _cached_eval(split.lower())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

