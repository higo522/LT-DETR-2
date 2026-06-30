from pathlib import Path
from PIL import Image
from lightly_train._task_models.task_model_helpers import load_model

images_dirs = [
    Path("/mnt/d/Riding Mountatin/segments/rgb/11/Elk/February 4/feb4_453/cropped"),
]


weights = "/home/higo522/lightly/experiments/rect_convnext-small(6.3)/exported_models/exported_best.pt"
threshold = 0.7
class_id = 1  # 0=Moose, 1=Elk, 2=Deer

model = load_model(weights).eval()

for images_dir in images_dirs:
    img_paths = sorted(p for p in images_dir.glob("*") if p.is_file())

    labels_dir = images_dir / "labels"
    out_dir = labels_dir / "obj_train_data"
    train_txt = labels_dir / "train.txt"
    obj_data = labels_dir / "obj.data"
    obj_names = labels_dir / "obj.names"

    out_dir.mkdir(parents=True, exist_ok=True)

    train_lines = []

    for img_path in img_paths:
        with Image.open(img_path) as im:
            w, h = im.size

        pred = model.predict(str(img_path), threshold=threshold)

        lines = []
        for x1, y1, x2, y2 in pred["bboxes"].cpu().tolist():
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        train_lines.append(f"data/obj_train_data/{img_path.name}")

    train_txt.write_text("\n".join(train_lines))
    obj_data.write_text("classes = 2\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
    obj_names.write_text("Moose\nElk\n")