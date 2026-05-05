import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_results_csv(results_csv: Path):
    import pandas as pd

    df = pd.read_csv(results_csv)
    return df


def _find_map_columns(df) -> Tuple[str, str]:
    # Ultralytics typically uses:
    #   metrics/mAP50(B)
    #   metrics/mAP50-95(B)
    cols = list(df.columns)
    map50_candidates = [c for c in cols if "mAP50" in c and "95" not in c]
    map5095_candidates = [c for c in cols if "mAP50-95" in c or ("mAP50" in c and "95" in c)]

    # Prefer exact known names if present.
    exact_map50 = "metrics/mAP50(B)"
    exact_map5095 = "metrics/mAP50-95(B)"
    if exact_map50 in cols:
        map50_col = exact_map50
    else:
        if not map50_candidates:
            raise KeyError(f"Cannot find mAP50 column in results.csv. Columns: {cols}")
        map50_col = map50_candidates[0]

    if exact_map5095 in cols:
        map5095_col = exact_map5095
    else:
        if not map5095_candidates:
            raise KeyError(f"Cannot find mAP50-95 column in results.csv. Columns: {cols}")
        map5095_col = map5095_candidates[0]

    return map50_col, map5095_col


def _select_best_epoch(df, map50_col: str, map5095_col: str) -> Dict:
    # Ultralytics usually includes a "fitness" column (higher is better).
    if "fitness" in df.columns:
        idx = df["fitness"].idxmax()
    else:
        # Fallback: maximize mAP50-95 first, then mAP50.
        tmp = df[[map5095_col, map50_col]].copy()
        tmp["__score__"] = tmp[map5095_col] * 1000.0 + tmp[map50_col]
        idx = tmp["__score__"].idxmax()

    row = df.loc[idx].to_dict()
    return row


@dataclass
class AttemptConfig:
    imgsz: int
    epochs: int
    batch: int
    lr0: float
    patience: int
    name_suffix: str


def _unique_name(base_dir: Path, name: str) -> str:
    candidate = name
    i = 1
    while (base_dir / candidate).exists():
        i += 1
        candidate = f"{name}_{i}"
    return candidate


def _train_once(
    *,
    YOLO,
    data_yaml: Path,
    weights: str,
    run_root: Path,
    attempt: AttemptConfig,
    device: str,
    workers: int,
) -> Tuple[Path, Path]:
    """
    Returns (run_dir, results_csv_path).
    """
    project_dir = run_root
    project_dir.mkdir(parents=True, exist_ok=True)

    run_name = attempt.name_suffix
    run_name = _unique_name(project_dir, run_name)
    from ultralytics import YOLO as _YOLO  # noqa: N811

    model = _YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=attempt.epochs,
        imgsz=attempt.imgsz,
        batch=attempt.batch,
        lr0=attempt.lr0,
        patience=attempt.patience,
        single_cls=True,
        project=str(project_dir),
        name=run_name,
        workers=workers,
        device=device,
        # Keep defaults (augmentments, optimizer) unless user edits.
    )

    run_dir = project_dir / run_name
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        # Ultralytics may sometimes put results.csv under a subfolder; best-effort fallback.
        candidates = list(run_dir.rglob("results.csv"))
        if not candidates:
            raise FileNotFoundError(f"Cannot find results.csv under: {run_dir}")
        results_csv = candidates[0]
    return run_dir, results_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\WEEK2")
    parser.add_argument("--dataset-out-root", default=r"D:\WEEK2\yolov8n_pig_dataset")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset; do not train.")
    parser.add_argument("--device", default="0", help="Ultralytics device string, e.g. '0' or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights", default="yolov8n.pt", help="Pretrained weights.")
    parser.add_argument("--best-model-out", default=r"D:\WEEK2\yolov8n_pig\best.pt")
    parser.add_argument("--excel-out", default=r"D:\WEEK2\yolov8n_pig\training_metrics.xlsx")
    parser.add_argument("--threshold-map50", type=float, default=0.96)
    parser.add_argument("--threshold-map50-95", type=float, default=0.76)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild dataset output folder.")
    args = parser.parse_args()

    dataset_out_root = Path(args.dataset_out_root)
    data_yaml = dataset_out_root / "data.yaml"
    if args.rebuild_dataset or not data_yaml.exists():
        from prepare_dataset import build_dataset  # type: ignore
        # Import locally without changing sys.path too much.

        # Build dataset by importing prepare_dataset.py.
        # We replicate its args with a small shim object.
        import argparse as _argparse

        ds_args = _argparse.Namespace(
            project_root=args.project_root,
            rawframes_dir=str(Path(args.project_root) / "rawframes"),
            label_root=str(Path(args.project_root) / "pig_detect_json_dataset" / "labels"),
            val_list_file=str(Path(args.project_root) / "val_images (1).txt"),
            out_root=str(dataset_out_root),
            rebuild=bool(args.rebuild_dataset),
            symlink_images=True,
            force_copy=False,
        )
        build_dataset(ds_args)

    if args.prepare_only:
        print("Dataset prepared only; skip training.")
        return

    # Lazy import so dataset-prep can run without ultralytics installed.
    from ultralytics import YOLO  # noqa: F401
    import pandas as pd

    # Attempt configurations (edit if needed).
    attempts: List[AttemptConfig] = [
        AttemptConfig(imgsz=640, epochs=200, batch=16, lr0=0.01, patience=50, name_suffix="train_attempt_1"),
        AttemptConfig(imgsz=800, epochs=250, batch=16, lr0=0.005, patience=70, name_suffix="train_attempt_2"),
        AttemptConfig(imgsz=960, epochs=300, batch=8, lr0=0.003, patience=90, name_suffix="train_attempt_3"),
    ][: args.max_attempts]

    run_root = dataset_out_root.parent / "yolov8n_pig_runs"

    summary_rows: List[Dict] = []
    best_overall: Optional[Dict] = None
    best_run_dir: Optional[Path] = None
    best_results_csv: Optional[Path] = None

    for i, attempt in enumerate(attempts, start=1):
        print(f"=== Attempt {i}/{len(attempts)}: {attempt.name_suffix} ===")
        run_dir, results_csv = _train_once(
            YOLO=YOLO,
            data_yaml=data_yaml,
            weights=args.weights,
            run_root=run_root,
            attempt=attempt,
            device=args.device,
            workers=args.workers,
        )

        df = _load_results_csv(results_csv)
        map50_col, map5095_col = _find_map_columns(df)
        best_row = _select_best_epoch(df, map50_col=map50_col, map5095_col=map5095_col)

        map50 = float(best_row[map50_col])
        map5095 = float(best_row[map5095_col])
        epoch = best_row.get("epoch", None)

        row = {
            "attempt": i,
            "run_dir": str(run_dir),
            "best_epoch": epoch,
            "best_mAP50": map50,
            "best_mAP50-95": map5095,
            "threshold_map50_ok": map50 >= args.threshold_map50,
            "threshold_map50-95_ok": map5095 >= args.threshold_map50_95,
        }
        summary_rows.append(row)

        print(f"Best mAP50={map50:.4f}, mAP50-95={map5095:.4f}")

        ok = (map50 >= args.threshold_map50) and (map5095 >= args.threshold_map50_95)
        if best_overall is None or (map5095 > best_overall["best_mAP50-95"]):
            best_overall = row
            best_run_dir = run_dir
            best_results_csv = results_csv

        if ok:
            print("Threshold achieved; stop early.")
            break

    # Pick best run for epoch-level sheet
    if best_run_dir is None or best_results_csv is None or best_overall is None:
        raise RuntimeError("No training run produced metrics.")

    best_df = _load_results_csv(best_results_csv)

    # Export excel
    excel_out = Path(args.excel_out)
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        best_df.to_excel(writer, sheet_name="best_run_epochs", index=False)

    # Export best model weights (best.pt)
    weights_src = best_run_dir / "weights" / "best.pt"
    if weights_src.exists():
        Path(args.best_model_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_src, Path(args.best_model_out))
        print(f"Copied best model to: {args.best_model_out}")
    else:
        print(f"Warning: best.pt not found at: {weights_src}")

    print(f"Excel exported to: {excel_out}")


if __name__ == "__main__":
    main()

