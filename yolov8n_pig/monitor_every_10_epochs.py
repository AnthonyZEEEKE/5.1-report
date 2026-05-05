import argparse
import time
from pathlib import Path


def read_last_metrics(results_csv: Path):
    # Read last non-empty line (csv has header)
    try:
        lines = results_csv.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = results_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    last = lines[-1].split(",")
    if len(last) != len(header):
        return None
    row = dict(zip(header, last))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=r"D:\WEEK2\yolov8n_pig_runs\train_attempt_1\results.csv",
        help="Path to Ultralytics results.csv",
    )
    parser.add_argument("--every", type=int, default=10, help="Report every N epochs")
    parser.add_argument("--poll-seconds", type=float, default=15.0, help="Polling interval")
    args = parser.parse_args()

    results = Path(args.results)
    print(f"Monitoring: {results}")

    last_reported = None
    while True:
        if results.exists():
            row = read_last_metrics(results)
            if row is not None and "epoch" in row:
                try:
                    epoch = int(float(row["epoch"]))
                except Exception:
                    epoch = None

                if epoch is not None:
                    if epoch > 0 and (epoch % args.every == 0) and (last_reported != epoch):
                        map50 = row.get("metrics/mAP50(B)", None)
                        map5095 = row.get("metrics/mAP50-95(B)", None)
                        p = row.get("metrics/precision(B)", None)
                        r = row.get("metrics/recall(B)", None)
                        print(
                            f"[epoch {epoch}] mAP50={map50} mAP50-95={map5095} P={p} R={r} (from results.csv)"
                        )
                        last_reported = epoch

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

