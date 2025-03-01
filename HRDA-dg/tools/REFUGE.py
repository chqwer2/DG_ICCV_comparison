from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd,
                              RandGaussianNoised, RandGaussianSmoothd, RandRotate90d, Compose, ResizeWithPadOrCropd)
import os
import glob

# Define paths
refuge_root = "/home/cbtil3/Downloads/REFUGE"
# refuge_crop_size = (512, 512)


# Define transforms
def get_transforms(train=True):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=refuge_crop_size),
        ScaleIntensityd(keys=["image"]),
    ]

    if train:
        transforms.extend([
            # RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=refuge_crop_size, pos=1, neg=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 2.0), sigma_y=(0.1, 2.0)),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0, std=10),
        ])

    return Compose(transforms)


# Function to load dataset
def load_dataset(split="train"):
    img_dir  = os.path.join(refuge_root, f"{split}/Images")
    mask_dir = os.path.join(refuge_root, f"{split}/Masks")

    img_files  = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    print(f"Found {len(img_files)} images and {len(mask_files)} masks in {split} split")

    data = [{"image": img, "label": mask} for img, mask in zip(img_files, mask_files)]
    return data

import os
import numpy as np
import json
import cv2
from tqdm import tqdm


def generate_samples_with_class():
    """
    Generates samples_with_class.json, mapping class IDs to image files containing them.

    Args:
        data_root (str): Root directory of the dataset.
        masks_dir (str): Subdirectory where segmentation masks are stored.
        class_labels (dict): Mapping of class names to pixel values.

    Returns:
        str: Path to the generated JSON file.
    """
    masks_path = "/home/cbtil3/Downloads/REFUGE/train/Masks"  # Change this to your REFUGE masks directory
    output_json_path = "./data/gta/samples_with_class.json"

    # Define class labels and their corresponding pixel values in masks
    class_labels = {
        "background": 0,
        "optic_disc": 1,
        "optic_cup": 2
    }
    min_pixels = 1000

    samples_with_class = {str(class_value): [] for class_value in class_labels.values()}

    mask_files = [f for f in os.listdir(masks_path) if f.endswith(".png") or f.endswith(".jpg")]

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Skipping {mask_file} (could not load)")
            continue

        # Find unique classes in this mask
        unique_classes, counts = np.unique(mask, return_counts=True)
        class_pixel_counts = dict(zip(unique_classes, counts))  # Map class to pixel count

        for class_value, pixel_count in class_pixel_counts.items():
            if str(class_value) in samples_with_class and pixel_count > min_pixels:
                samples_with_class[str(class_value)].append([mask_file, int(pixel_count)])  # Store as list of lists

    # Save results to JSON
    with open(output_json_path, "w") as f:
        json.dump(samples_with_class, f, indent=4)

    print(f"Samples-with-class mapping saved to {output_json_path}")
    return output_json_path


def write_sample_class_stats():

    # Define dataset path
    masks_path = "/home/cbtil3/Downloads/REFUGE/train/Masks"  # Change this to your REFUGE masks directory
    output_json_path = "./data/gta/sample_class_stats.json"

    # Define class labels and their corresponding pixel values in masks
    class_labels = {
        "background": 0,
        "optic_disc": 1,
        "optic_cup": 2
    }

    class_pixel_counts = []

    mask_files = [f for f in os.listdir(masks_path) if f.endswith(".png") or f.endswith(".jpg")]

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Skipping {mask_file} (could not load)")
            continue

        file_class_counts = {"file": mask_file}  # Store per-file class stats

        # Count pixels for each class
        for class_name, class_value in class_labels.items():
            count = int(np.sum(mask == class_value))
            file_class_counts[str(class_value)] = count  # Store as string for JSON compatibility

        class_pixel_counts.append(file_class_counts)

    # Save the statistics to JSON
    with open(output_json_path, "w") as f:
        json.dump(class_pixel_counts, f, indent=4)

    print(f"Class statistics saved to {output_json_path}")
    return output_json_path


if __name__ == "__main__":
    write_sample_class_stats()
    generate_samples_with_class()

    # Create datasets
    data_train = load_dataset("train")
    data_val   = load_dataset("val")
    data_test  = load_dataset("test")

    # Create MONAI datasets
    dataset_train = CacheDataset(data=data_train, transform=get_transforms(train=True))
    dataset_val   = Dataset(data=data_val, transform=get_transforms(train=False))
    dataset_test  = Dataset(data=data_test, transform=get_transforms(train=False))

    # Create dataloaders
    train_loader = DataLoader(dataset_train, batch_size=8, num_workers=8, pin_memory=True, shuffle=True)
    val_loader   = DataLoader(dataset_val, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)
    test_loader  = DataLoader(dataset_test, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)




