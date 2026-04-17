from pathlib import Path
import shutil

# ============================================================
# WaRP-D class remapping for grouped recyclable categories
#
# New grouped classes:
# 0 = plastic
# 1 = detergent
# 2 = can
# 3 = cardboard
# 4 = glass
# 5 = canister
#
# Original WaRP-D classes.txt indices:
# 0  bottle-blue
# 1  bottle-green
# 2  bottle-dark
# 3  bottle-milk
# 4  bottle-transp
# 5  bottle-multicolor
# 6  bottle-yogurt
# 7  bottle-oil
# 8  cans
# 9  juice-cardboard
# 10 milk-cardboard
# 11 detergent-color
# 12 detergent-transparent
# 13 detergent-box
# 14 canister
# 15 bottle-blue-full
# 16 bottle-transp-full
# 17 bottle-dark-full
# 18 bottle-green-full
# 19 bottle-multicolorv-full
# 20 bottle-milk-full
# 21 bottle-oil-full
# 22 detergent-white
# 23 bottle-blue5l
# 24 bottle-blue5l-full
# 25 glass-transp
# 26 glass-dark
# 27 glass-green
# ============================================================

CLASS_MAP = {
    # Plastic bottles and plastic containers
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    23: 0,
    24: 0,

    # Detergent containers
    11: 1,
    12: 1,
    13: 1,
    22: 1,

    # Metal cans
    8: 2,

    # Cardboard / cartons
    9: 3,
    10: 3,

    # Glass
    25: 4,
    26: 4,
    27: 4,

    # Canister
    14: 5
}

# Valid image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def remap_label_file(src_label_path: Path, dst_label_path: Path) -> bool:
    """
    Read one YOLO label file, convert original WaRP-D class IDs
    into grouped class IDs, and write the remapped annotations.

    Returns:
        True if the output file contains at least one valid object.
        False if no valid objects remain after filtering.
    """
    remapped_lines = []

    with src_label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            # YOLO annotation format:
            # class_id x_center y_center width height
            if len(parts) != 5:
                continue

            old_class = int(parts[0])
            bbox = parts[1:]

            # Skip classes we do not want in the reduced dataset
            if old_class not in CLASS_MAP:
                continue

            new_class = CLASS_MAP[old_class]
            remapped_lines.append(f"{new_class} {' '.join(bbox)}")

    # Make sure output folder exists
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    # Write remapped label file
    with dst_label_path.open("w", encoding="utf-8") as f:
        for line in remapped_lines:
            f.write(line + "\n")

    return len(remapped_lines) > 0


def copy_matching_images(src_image_dir: Path, dst_image_dir: Path, valid_stems: set[str]) -> None:
    """
    Copy only images whose labels still contain valid remapped objects.
    """
    dst_image_dir.mkdir(parents=True, exist_ok=True)

    for image_path in src_image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if image_path.stem not in valid_stems:
            continue

        shutil.copy2(image_path, dst_image_dir / image_path.name)


def process_split(src_images_dir: Path, src_labels_dir: Path, dst_images_dir: Path, dst_labels_dir: Path) -> None:
    """
    Process one dataset split (train or test):
    - remap label files
    - keep only images with valid remapped labels
    """
    valid_stems = set()

    for label_path in src_labels_dir.glob("*.txt"):
        dst_label_path = dst_labels_dir / label_path.name
        has_objects = remap_label_file(label_path, dst_label_path)

        if has_objects:
            valid_stems.add(label_path.stem)
        else:
            # Delete empty output labels if nothing remained
            if dst_label_path.exists():
                dst_label_path.unlink()

    copy_matching_images(src_images_dir, dst_images_dir, valid_stems)

    print(f"Processed {len(valid_stems)} valid files from {src_images_dir}")


def main() -> None:
    """
    Convert raw WaRP-D dataset into a grouped YOLO dataset.

    Expected raw structure:
        WaRP/Warp-D/train/images
        WaRP/Warp-D/train/labels
        WaRP/Warp-D/test/images
        WaRP/Warp-D/test/labels

    Output structure:
        data/processed/warp_recycle/images/train
        data/processed/warp_recycle/labels/train
        data/processed/warp_recycle/images/test
        data/processed/warp_recycle/labels/test
    """
    base_src = Path("WaRP/Warp-D")
    base_dst = Path("data/processed/warp_recycle")

    splits = ["train", "test"]

    for split in splits:
        src_images_dir = base_src / split / "images"
        src_labels_dir = base_src / split / "labels"

        dst_images_dir = base_dst / "images" / split
        dst_labels_dir = base_dst / "labels" / split

        process_split(
            src_images_dir=src_images_dir,
            src_labels_dir=src_labels_dir,
            dst_images_dir=dst_images_dir,
            dst_labels_dir=dst_labels_dir
        )

    print("Class remapping complete.")


if __name__ == "__main__":
    main()