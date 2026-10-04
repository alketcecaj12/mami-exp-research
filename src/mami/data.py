
"""Load images from explicit split folders and check content overlap."""
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image, ImageOps


def load_splits(config):
    settings = config["data"]
    root = Path(settings["root"]).expanduser().resolve()
    folders = settings["image_dirs"]
    splits, report = {}, {}
    seen = {}

    # Hold out test first, then validation, then retain training examples.
    for split in ("test", "validation", "train"):
        folder = (root / folders[split]).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image folder: {folder}")

        frame = pd.read_csv(
            root / settings[split],
            sep="\t",
            keep_default_na=False,
        )
        required = {"file_name", "label", "text"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{split}: missing required columns")
        if frame.file_name.duplicated().any():
            raise ValueError(f"{split}: duplicate metadata filenames")

        frame["label"] = pd.to_numeric(frame.label, errors="raise")
        if not frame.label.isin([0, 1]).all():
            raise ValueError(f"{split}: labels must be 0 or 1")

        kept, missing, excluded = [], [], []
        for row in frame.to_dict("records"):
            image_path = (folder / str(row["file_name"])).resolve()
            if not image_path.is_relative_to(folder):
                raise ValueError(f"Image path outside folder: {image_path}")
            if not image_path.is_file():
                missing.append(row["file_name"])
                continue

            # Hash decoded pixels: also catches identical pixels stored
            # with different file metadata. Not a near-duplicate detector.
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                digest = hashlib.sha256(
                    str(image.size).encode() + image.tobytes()
                ).hexdigest()

            previous = seen.get(digest)
            if previous and previous["split"] != split:
                if previous["label"] != int(row["label"]):
                    raise ValueError(
                        f"Identical image has conflicting labels: "
                        f"{previous['path']} and {image_path}"
                    )
                excluded.append({
                    "file_name": row["file_name"],
                    "retained_split": previous["split"],
                    "retained_image": previous["path"],
                })
                continue

            if previous is None:
                seen[digest] = {
                    "split": split,
                    "label": int(row["label"]),
                    "path": str(image_path),
                }

            row["image_path"] = str(image_path)
            kept.append(row)

        if missing and settings.get("missing_images", "error") == "error":
            raise FileNotFoundError(
                f"{split}: {len(missing)} missing images in {folder}. "
                f"First missing filenames: {missing[:5]}"
            )

        if not kept:
            raise ValueError(f"{split}: no usable examples")

        selected = pd.DataFrame(kept)
        splits[split] = selected
        report[split] = {
            "metadata_rows": len(frame),
            "available_rows": len(selected),
            "missing_images": len(missing),
            "removed_content_overlap": len(excluded),
            "excluded_examples": excluded,
            "class_counts": {
                str(k): int(v)
                for k, v in selected.label.value_counts().items()
            },
        }

    return splits, report
