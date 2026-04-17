from pathlib import Path
import random
import shutil

# Fraction of training data to move into validation
VAL_RATIO = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Valid image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files(images_dir: Path) -> list[Path]:
    """
    Return all valid image files in a directory.
    """
    return [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def move_image_and_label(
    image_path: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path
) -> bool:
    """
    Move one image and its matching YOLO label file from train to val.

    Returns:
        bool: True if both image and label were moved successfully,
              False if the matching label was missing.
    """
    label_path = src_labels_dir / f"{image_path.stem}.txt"

    # Skip images that do not have a matching label
    if not label_path.exists():
        print(f"Skipping {image_path.name}: missing label file.")
        return False

    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    shutil.move(str(image_path), str(dst_images_dir / image_path.name))
    shutil.move(str(label_path), str(dst_labels_dir / label_path.name))

    return True


def main() -> None:
    """
    Split a portion of the processed training set into a validation set.

    Expected input structure:
        data/processed/warp_recycle/images/train
        data/processed/warp_recycle/labels/train

    Output structure after split:
        data/processed/warp_recycle/images/train
        data/processed/warp_recycle/images/val
        data/processed/warp_recycle/labels/train
        data/processed/warp_recycle/labels/val
    """
    random.seed(RANDOM_SEED)

    base_dir = Path("data/processed/warp_recycle")

    train_images_dir = base_dir / "images" / "train"
    train_labels_dir = base_dir / "labels" / "train"

    val_images_dir = base_dir / "images" / "val"
    val_labels_dir = base_dir / "labels" / "val"

    # Safety checks
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Training images folder not found: {train_images_dir}")

    if not train_labels_dir.exists():
        raise FileNotFoundError(f"Training labels folder not found: {train_labels_dir}")

    image_files = get_image_files(train_images_dir)

    if not image_files:
        raise ValueError("No training images found to split.")

    # Shuffle image list and choose validation subset
    random.shuffle(image_files)
    val_count = int(len(image_files) * VAL_RATIO)

    if val_count == 0:
        raise ValueError("Validation split produced 0 files. Increase dataset size or VAL_RATIO.")

    val_images = image_files[:val_count]

    moved_count = 0

    for image_path in val_images:
        moved = move_image_and_label(
            image_path=image_path,
            src_labels_dir=train_labels_dir,
            dst_images_dir=val_images_dir,
            dst_labels_dir=val_labels_dir
        )
        if moved:
            moved_count += 1

    print(f"Moved {moved_count} image-label pairs to validation set.")
    print(f"Remaining training images: {len(get_image_files(train_images_dir))}")
    print(f"Validation images: {len(get_image_files(val_images_dir))}")


if __name__ == "__main__":
    main()