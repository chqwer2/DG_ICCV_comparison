from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd,
                              RandGaussianNoised, RandGaussianSmoothd, Resized, RandRotate90d, Compose, ResizeWithPadOrCropd)
import os
import glob

# Define paths
refuge_root = "/home/cbtil3/Downloads/REFUGE"
refuge_root = "/bask/projects/j/jiaoj-rep-learn/Hao/datasets/REFUGE"
refuge_crop_size = (512, 512)


# Define transforms
def get_transforms(train=True):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=refuge_crop_size),
        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=refuge_crop_size),
        ScaleIntensityd(keys=["image"]),
    ]

    if train:
        transforms.extend([
            # RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
            #                        spatial_size=refuge_crop_size, pos=1, neg=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 2.0), sigma_y=(0.1, 2.0)),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0, std=10),
        ])

    return Compose(transforms)


# Function to load dataset
def load_dataset(args, split="train"):
    img_dir  = os.path.join(refuge_root, f"{split}/Images")
    mask_dir = os.path.join(refuge_root, f"{split}/Masks")

    img_files  = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    print(f"Found {len(img_files)} images and {len(mask_files)} masks in {split} split")

    data = [{"image": img, "label": mask} for img, mask in zip(img_files, mask_files)]


    # data = data[:10]

    if split == "train":
        trainset = CacheDataset(data=data, transform=get_transforms(train=True))
        loader = DataLoader(trainset, batch_size=args.batch_size,
                            num_workers=args.data_loader_workers, pin_memory=True, shuffle=True)
        return loader, trainset

    elif split == "val":
        set = Dataset(data=data, transform=get_transforms(train=False))
        loader = DataLoader(set, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)


    elif split == "test":
        set = Dataset(data=data, transform=get_transforms(train=False))
        loader = DataLoader(set, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)

    return loader, set







if __name__ == "__main__":
    # Create datasets
    data_train = load_dataset("train")
    data_val   = load_dataset("val")
    data_test  = load_dataset("test")


