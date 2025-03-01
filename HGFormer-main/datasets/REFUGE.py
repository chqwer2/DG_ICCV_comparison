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







if __name__ == "__main__":
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




