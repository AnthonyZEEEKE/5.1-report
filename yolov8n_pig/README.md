# YOLOv8n Pig Detection (1 class)

This folder contains scripts to:
1. Prepare a YOLOv8 dataset from your existing `rawframes/` and `pig_detect_json_dataset/labels/`.
2. Train `yolov8n` on a single class (`pig`) and export `mAP50` / `mAP50-95` to Excel.
3. Run inference with the exported `best.pt`.

## Files used (from `D:\WEEK2`)
- Images: `rawframes/`
- Labels (YOLO txt): `pig_detect_json_dataset/labels/train` and `.../labels/val`
- Usable val subset list: `val_images (1).txt`

## Requirements
Install dependencies (example):
```bash
pip install ultralytics pandas openpyxl
```
(Ultralytics will install the needed PyTorch stack for your environment.)

## 1) Prepare dataset
```bash
python prepare_dataset.py --rebuild
```
Outputs:
- `D:\WEEK2\yolov8n_pig_dataset\data.yaml`
- Symlinked/copied `images/train` and `images/val`
- `dataset_stats.json` in the dataset folder

## 2) Train and export Excel
```bash
python auto_train_and_export_excel.py --device 0 --rebuild-dataset
```
Outputs:
- `D:\WEEK2\yolov8n_pig\training_metrics.xlsx`
- `D:\WEEK2\yolov8n_pig\best.pt`
- Training logs under `D:\WEEK2\yolov8n_pig_runs\...`

The script tries up to 3 training attempts and stops early if:
- `mAP50 >= 0.96`
- `mAP50-95 >= 0.76`

## 3) Inference (detect pig)
```bash
python infer.py --source "D:\WEEK2\rawframes\MyVideo_074 - Trim" --imgsz 640 --conf 0.25
```
Outputs annotated images to `D:\WEEK2\yolov8n_pig_infer_out\...`

