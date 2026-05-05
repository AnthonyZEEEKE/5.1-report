import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


def _as_posix(p: Path) -> str:
    return p.as_posix()


def try_symlink_file(src: Path, dst: Path) -> bool:
    """
    Try to create a symlink from dst -> src.
    Returns True if symlink created, False if fallback copy was used.
    """
    try:
        if dst.exists():
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(src), str(dst))
        return True
    except Exception:
        # Fallback to copy if symlink is not permitted.
        shutil.copy2(src, dst)
        return False


def link_or_copy_dir(src_dir: Path, dst_dir: Path) -> None:
    """
    Link directories when possible; otherwise copy.
    """
    if dst_dir.exists():
        return
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src_dir), str(dst_dir), target_is_directory=True)
    except Exception:
        shutil.copytree(src_dir, dst_dir)


def count_label_boxes(label_file: Path) -> int:
    """
    YOLO label format per line:
      class x_center y_center width height
    We count valid lines (5 tokens).
    """
    if not label_file.exists():
        return 0
    try:
        cnt = 0
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    cnt += 1
        return cnt
    except Exception:
        return 0


def extract_val_relpath_from_list_line(line: str) -> Optional[str]:
    """
    Expected line formats (observed in this project):
      pig_dataset/images/val/<subpath>/img_xxx.jpg
    """
    s = line.strip().replace("\\", "/")
    if not s:
        return None
    marker = "images/val/"
    if marker in s:
        rel = s.split(marker, 1)[1]
        return rel.lstrip("/")
    # Fallback: try to detect ".../val/<subpath>.jpg" (best-effort).
    # If this fails, caller will skip.
    return None


def build_dataset(args: argparse.Namespace) -> Dict[str, int]:
    project_root = Path(args.project_root)
    rawframes_dir = Path(args.rawframes_dir)
    label_root = Path(args.label_root)
    val_list_file = Path(args.val_list_file)

    out_root = Path(args.out_root)
    out_images_train = out_root / "images" / "train"
    out_images_val = out_root / "images" / "val"
    out_labels_train = out_root / "labels" / "train"
    out_labels_val = out_root / "labels" / "val"

    if args.rebuild and out_root.exists():
        shutil.rmtree(out_root)

    out_images_train.mkdir(parents=True, exist_ok=True)
    out_images_val.mkdir(parents=True, exist_ok=True)

    # Labels are already in YOLO format and are small: link whole directories.
    link_or_copy_dir(label_root / "train", out_labels_train)
    link_or_copy_dir(label_root / "val", out_labels_val)

    # Stats
    stats = {
        "train_images_used": 0,
        "val_images_used": 0,
        "train_labels_used": 0,
        "val_labels_used": 0,
        "train_boxes_total": 0,
        "val_boxes_total": 0,
    }

    # --------------------
    # Build train split from labels/train/*
    # --------------------
    labels_train_dir = label_root / "train"
    for label_txt in labels_train_dir.rglob("*.txt"):
        rel_txt = label_txt.relative_to(labels_train_dir)  # subpath/.../img_xxx.txt
        rel_img = rel_txt.with_suffix(".jpg")
        src_img = rawframes_dir / rel_img
        if not src_img.exists():
            continue
        boxes = count_label_boxes(label_txt)
        if boxes <= 0:
            continue

        dst_img = out_images_train / rel_img
        if args.symlink_images and not args.force_copy:
            try_symlink_file(src_img, dst_img)
        else:
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

        stats["train_images_used"] += 1
        stats["train_labels_used"] += 1
        stats["train_boxes_total"] += boxes

    # --------------------
    # Build val split from val_list_file (usable subset)
    # --------------------
    if not val_list_file.exists():
        raise FileNotFoundError(f"val_list_file not found: {val_list_file}")

    with val_list_file.open("r", encoding="utf-8") as f:
        for line in f:
            rel_img_str = extract_val_relpath_from_list_line(line)
            if not rel_img_str:
                continue
            rel_img = Path(rel_img_str)  # subpath/.../img_xxx.jpg
            src_img = rawframes_dir / rel_img
            if not src_img.exists():
                continue

            label_txt = label_root / "val" / rel_img.with_suffix(".txt")
            boxes = count_label_boxes(label_txt)
            if boxes <= 0:
                continue

            dst_img = out_images_val / rel_img
            if args.symlink_images and not args.force_copy:
                try_symlink_file(src_img, dst_img)
            else:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

            stats["val_images_used"] += 1
            stats["val_labels_used"] += 1
            stats["val_boxes_total"] += boxes

    # Write data.yaml
    data_yaml = out_root / "data.yaml"
    data_yaml_lines = [
        f"path: {_as_posix(out_root.resolve())}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: pig",
        "",
    ]
    data_yaml.write_text("\n".join(data_yaml_lines), encoding="utf-8")

    (out_root / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\WEEK2")
    parser.add_argument("--rawframes-dir", default=r"D:\WEEK2\rawframes")
    parser.add_argument("--label-root", default=r"D:\WEEK2\pig_detect_json_dataset\labels")
    parser.add_argument("--val-list-file", default=r"D:\WEEK2\val_images (1).txt")
    parser.add_argument("--out-root", default=r"D:\WEEK2\yolov8n_pig_dataset")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild dataset output folder")
    parser.add_argument("--symlink-images", action="store_true", default=True, help="Symlink images (fallback to copy)")
    parser.add_argument("--force-copy", action="store_true", help="Always copy images instead of symlink")
    args = parser.parse_args()

    stats = build_dataset(args)
    print("Dataset prepared.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

