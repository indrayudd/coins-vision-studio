from __future__ import annotations

"""
main API file for coin thing.

Yes this is intentionally over-commented and kinda casual because this is
me documenting every little trick for project grading / future me panic mode.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .coin_pipeline import draw_many_and_process, estimate_coin_count, get_image_paths

# project paths we keep reusing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
GROUND_TRUTHS_PATH = DATASET_ROOT / "ground_truths.txt"

app = FastAPI(
    title="Coin Counter API",
    description="Classical edge/boundary coin counting with explainable step-by-step outputs.",
    version="1.0.0",
)


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# keep frontend local hosts open so vite app can call API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Simple alive check used by frontend + debugging scripts."""
    return {"status": "ok"}


def _normalize_dataset_path(path_text: str) -> str:
    """Normalize any path string into a canonical lowercase `dataset/...` shape."""
    fixed = path_text.strip().replace("\\", "/").lower()
    marker = "dataset/"
    idx = fixed.find(marker)
    if idx >= 0:
        return fixed[idx:]
    return fixed


@lru_cache(maxsize=1)
def _load_ground_truths() -> dict[str, int]:
    """Read and cache `ground_truths.txt` as {normalized_path: coin_count}."""
    if not GROUND_TRUTHS_PATH.exists():
        return {}

    gt_map: dict[str, int] = {}
    for raw_line in GROUND_TRUTHS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        path_part, count_part = line.rsplit(",", 1)
        try:
            gt_count = int(count_part.strip())
        except ValueError:
            # junk line, ignore it quietly.
            continue
        gt_map[_normalize_dataset_path(path_part)] = gt_count
    return gt_map


def _attach_ground_truth(item: dict, ground_truths: dict[str, int]) -> dict:
    """Attach ground-truth count and signed/absolute error for one draw item."""
    item_copy = deepcopy(item)
    key = _normalize_dataset_path(item_copy.get("image_path", ""))
    gt_val = ground_truths.get(key)
    item_copy["ground_truth_count"] = gt_val

    if gt_val is None:
        item_copy["error"] = None
        item_copy["absolute_error"] = None
        return item_copy

    err = int(item_copy["coin_count"]) - int(gt_val)
    item_copy["error"] = err
    item_copy["absolute_error"] = abs(err)
    return item_copy


def _evaluate_split(split: str) -> dict:
    """Run full-dataset evaluation for one split and return aggregate metrics."""
    img_paths = get_image_paths(dataset_root=DATASET_ROOT, split=split)
    if not img_paths:
        raise ValueError(f"No images found in split '{split}' under {DATASET_ROOT}")

    gt_map = _load_ground_truths()
    if not gt_map:
        raise ValueError(f"No valid ground-truth records found in {GROUND_TRUTHS_PATH}")

    # parallel count estimation so evaluation doesn't take forever.
    rows: list[dict] = []
    worker_count = min(8, max(1, len(img_paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool_thing:
        future_to_path = {pool_thing.submit(estimate_coin_count, path): path for path in img_paths}
        for future in as_completed(future_to_path):
            img_path = future_to_path[future]
            pred_count = int(future.result())
            rel_path = str(img_path.relative_to(DATASET_ROOT.parent))
            gt_val = gt_map.get(_normalize_dataset_path(rel_path))
            rows.append(
                {
                    "image_path": rel_path,
                    "predicted_count": pred_count,
                    "ground_truth_count": gt_val,
                    "error": None if gt_val is None else pred_count - int(gt_val),
                }
            )

    rows.sort(key=lambda row: row["image_path"])
    matched_rows = [row for row in rows if row["ground_truth_count"] is not None]
    if not matched_rows:
        raise ValueError("No matching ground-truth labels were found for this split.")

    total_pred = int(sum(int(row["predicted_count"]) for row in matched_rows))
    total_gt = int(sum(int(row["ground_truth_count"]) for row in matched_rows))
    abs_errs = [abs(int(row["error"])) for row in matched_rows]
    sq_errs = [int(row["error"]) ** 2 for row in matched_rows]
    exact_hits = sum(1 for row in matched_rows if int(row["error"]) == 0)
    n_scored = len(matched_rows)

    top_errors = sorted(matched_rows, key=lambda row: abs(int(row["error"])), reverse=True)[:10]

    return {
        "split": split,
        "num_images": len(rows),
        "num_scored": n_scored,
        "num_missing_ground_truth": len(rows) - n_scored,
        "total_predicted": total_pred,
        "total_ground_truth": total_gt,
        "total_error": total_pred - total_gt,
        "total_absolute_error": int(sum(abs_errs)),
        "mae": float(sum(abs_errs) / n_scored),
        "rmse": float((sum(sq_errs) / n_scored) ** 0.5),
        "mean_error": float(sum(int(row["error"]) for row in matched_rows) / n_scored),
        "exact_match_rate": float(exact_hits / n_scored),
        "top_errors": top_errors,
    }


@lru_cache(maxsize=12)
def _cached_evaluation(split: str) -> dict:
    """Memoized evaluation so repeated UI refresh is fast."""
    return _evaluate_split(split)


@app.get("/api/splits")
def splits() -> dict[str, list[str]]:
    """List dataset folders plus the synthetic `all` split."""
    subdirs = sorted([p.name for p in DATASET_ROOT.iterdir() if p.is_dir()])
    return {"splits": ["all", *subdirs]}


@app.get("/api/draw")
def draw(
    split: str = Query(default="all", description="Dataset split/folder to draw from."),
    seed: int | None = Query(default=None, description="Optional RNG seed for reproducible draws."),
    count: int = Query(default=10, ge=1, le=25, description="Number of random samples to draw."),
) -> dict:
    """Draw random image batch, run pipeline, attach ground-truth and per-sample error."""
    try:
        # extra guard so typo split gives clean 404.
        if split != "all":
            available_splits = {p.parent.name.lower() for p in get_image_paths(DATASET_ROOT, "all")}
            if split.lower() not in available_splits:
                raise HTTPException(status_code=404, detail=f"Split '{split}' not found in dataset.")

        gt_map = _load_ground_truths()
        batch_items = draw_many_and_process(dataset_root=DATASET_ROOT, split=split, seed=seed, count=count)
        batch_items = [_attach_ground_truth(item, gt_map) for item in batch_items]
        return {"count": len(batch_items), "items": batch_items, "split": split, "seed": seed}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc


@app.get("/api/evaluation")
def evaluation(
    split: str = Query(default="all", description="Dataset split/folder to evaluate."),
    refresh: bool = Query(default=False, description="Recompute metrics instead of using cached results."),
) -> dict:
    """Evaluate one split against ground truth and return dataset-level error stats."""
    try:
        split_lower = split.lower()
        if split_lower != "all":
            available_splits = {p.parent.name.lower() for p in get_image_paths(DATASET_ROOT, "all")}
            if split_lower not in available_splits:
                raise HTTPException(status_code=404, detail=f"Split '{split}' not found in dataset.")

        if refresh:
            _cached_evaluation.cache_clear()

        return _cached_evaluation(split_lower)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

