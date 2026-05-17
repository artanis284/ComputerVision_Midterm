import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install with `pip install ultralytics`."
        ) from exc
    return YOLO


def init_wandb(project: str, run_name: str | None, config: dict):
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Missing dependency: wandb. Install with `pip install wandb`.") from exc
    return wandb.init(project=project, name=run_name, config=config)


def find_split_dir(root: Path, split: str) -> Path | None:
    split_name = f"VisDrone2019-DET-{split}"
    split_dir = root / split_name
    nested_dir = split_dir / split_name
    if (split_dir / "images").exists() and (split_dir / "annotations").exists():
        return split_dir
    if (nested_dir / "images").exists() and (nested_dir / "annotations").exists():
        return nested_dir
    return None


def convert_visdrone_split(root: Path, split: str) -> Path | None:
    split_dir = find_split_dir(root, split)
    if split_dir is None:
        print(f"Skip {split}: images/annotations not found")
        return None
    image_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    label_dir = split_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for ann_path in ann_dir.glob("*.txt"):
        image_path = image_dir / f"{ann_path.stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        yolo_lines = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x, y, bw, bh, score, cls, trunc, occ = map(float, line.split(",")[:8])
            cls = int(cls)
            if cls < 1 or cls > 10 or bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            yolo_lines.append(f"{cls - 1} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (label_dir / f"{ann_path.stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
    return split_dir


def relative_image_dir(root: Path, split_dir: Path) -> str:
    return (split_dir.relative_to(root) / "images").as_posix()


def write_dataset_yaml(root: Path, out_yaml: Path, split_dirs: dict[str, Path]) -> None:
    text = f"""path: {root.as_posix()}
train: {relative_image_dir(root, split_dirs["train"])}
val: {relative_image_dir(root, split_dirs["val"])}
test: {relative_image_dir(root, split_dirs["test-dev"])}
names:
"""
    for i, name in enumerate(VISDRONE_NAMES):
        text += f"  {i}: {name}\n"
    out_yaml.write_text(text, encoding="utf-8")


def prepare_dataset(args):
    root = args.visdrone_root
    split_dirs = {}
    for split in ["train", "val", "test-dev"]:
        split_dir = convert_visdrone_split(root, split)
        if split_dir is not None:
            split_dirs[split] = split_dir
    missing = {"train", "val", "test-dev"} - set(split_dirs)
    if missing:
        raise SystemExit(f"Missing VisDrone splits: {sorted(missing)}")
    write_dataset_yaml(root, args.yaml, split_dirs)
    print(f"Wrote {args.yaml}")


def train_detector(args):
    YOLO = require_ultralytics()
    model = YOLO(args.model)
    model.train(
        data=str(args.yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
    )
    if args.wandb:
        save_dir = Path(model.trainer.save_dir)
        log_yolo_results_to_wandb(save_dir / "results.csv", args, save_dir)


def log_yolo_results_to_wandb(results_csv: Path, args, save_dir: Path) -> None:
    if not results_csv.exists():
        print(f"Warning: YOLO results file not found, skip wandb logging: {results_csv}")
        return
    wandb_run = init_wandb(
        args.wandb_project,
        args.wandb_run_name or args.name,
        {
            "yaml": str(args.yaml),
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "save_dir": str(save_dir),
        },
    )
    with results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row["epoch"]))
            metrics = {
                "epoch": epoch,
                "train/box_loss": float(row["train/box_loss"]),
                "train/cls_loss": float(row["train/cls_loss"]),
                "train/dfl_loss": float(row["train/dfl_loss"]),
                "val/precision": float(row["metrics/precision(B)"]),
                "val/recall": float(row["metrics/recall(B)"]),
                "val/mAP50": float(row["metrics/mAP50(B)"]),
                "val/mAP50-95": float(row["metrics/mAP50-95(B)"]),
                "val/box_loss": float(row["val/box_loss"]),
                "val/cls_loss": float(row["val/cls_loss"]),
                "val/dfl_loss": float(row["val/dfl_loss"]),
            }
            wandb_run.log(metrics, step=epoch)
    wandb_run.finish()


def resolve_weights(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("runs").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if len(candidates) == 1:
        print(f"Warning: {path} not found. Using detected checkpoint: {candidates[0]}")
        return candidates[0]
    if candidates:
        message = "\n".join(f"  {p}" for p in candidates[-5:])
        raise SystemExit(f"Checkpoint not found: {path}\nAvailable best.pt checkpoints:\n{message}")
    raise SystemExit(f"Checkpoint not found: {path}\nNo best.pt found under runs/.")


def side_of_line(point, line):
    (x1, y1), (x2, y2) = line
    return np.sign((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1))


def track_video(args):
    YOLO = require_ultralytics()
    args.weights = resolve_weights(args.weights)
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "tracked_counted.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    line = ((args.line[0], args.line[1]), (args.line[2], args.line[3]))
    last_side = {}
    counted_ids = set()
    frame_idx = 0
    occlusion_frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        annotated = results[0].plot()
        cv2.line(annotated, line[0], line[1], (0, 255, 255), 2)
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            for box, track_id in zip(xyxy, ids):
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                current_side = side_of_line((cx, cy), line)
                previous_side = last_side.get(track_id)
                if previous_side is not None and current_side != 0 and previous_side != 0:
                    if current_side != previous_side and track_id not in counted_ids:
                        counted_ids.add(track_id)
                if current_side != 0:
                    last_side[track_id] = current_side
                cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"Line count: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3,
        )
        writer.write(annotated)
        if args.occlusion_start <= frame_idx < args.occlusion_start + args.occlusion_frames:
            image_path = args.out_dir / f"occlusion_{frame_idx:06d}.jpg"
            cv2.imwrite(str(image_path), annotated)
            occlusion_frames.append(str(image_path))
        frame_idx += 1

    cap.release()
    writer.release()
    summary = {
        "video": str(args.video),
        "output": str(out_path),
        "line": args.line,
        "line_count": len(counted_ids),
        "occlusion_frames": occlusion_frames,
    }
    (args.out_dir / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Task 2: VisDrone YOLOv8 detection and tracking")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Convert VisDrone labels and write YAML")
    p.add_argument("--visdrone-root", type=Path, required=True)
    p.add_argument("--yaml", type=Path, default=Path("visdrone.yaml"))
    p.set_defaults(func=prepare_dataset)

    p = sub.add_parser("train", help="Fine-tune YOLOv8 on VisDrone")
    p.add_argument("--yaml", type=Path, default=Path("visdrone.yaml"))
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", type=Path, default=Path("runs/task2"))
    p.add_argument("--name", default="visdrone_yolov8n")
    p.add_argument("--wandb", action="store_true", help="Upload YOLO results.csv curves to wandb")
    p.add_argument("--wandb-project", default="cv-midterm-task2")
    p.add_argument("--wandb-run-name", default=None)
    p.set_defaults(func=train_detector)

    p = sub.add_parser("track", help="Track a 10-30 second video and count line crossings")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/task2/track"))
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--line", nargs=4, type=int, default=[100, 360, 1180, 360])
    p.add_argument("--occlusion-start", type=int, default=0)
    p.add_argument("--occlusion-frames", type=int, default=4)
    p.set_defaults(func=track_video)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
