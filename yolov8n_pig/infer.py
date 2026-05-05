import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=r"D:\WEEK2\yolov8n_pig\best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Image path or folder path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save-dir", default=r"D:\WEEK2\yolov8n_pig_infer_out")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Only class 0 ('pig').
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=[0],
        save=True,
        save_txt=False,
        project=str(save_dir),
        name="pred",
    )

    # Print a minimal summary for console usage.
    total = 0
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        total += len(r.boxes)
    print(f"Done. Total predicted boxes: {total}")


if __name__ == "__main__":
    main()

