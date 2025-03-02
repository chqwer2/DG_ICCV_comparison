import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm


def dice_coefficient(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou





# Load model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def test_model(model, dataloader, device):
    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Sigmoid for binary segmentation
            preds = (outputs > 0.5).float()  # Convert to binary mask

            for pred, mask in zip(preds, masks):
                dice = dice_coefficient(pred, mask)
                iou = iou_score(pred, mask)
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    print(f'Average DICE: {mean_dice:.4f}')
    print(f'Average IoU: {mean_iou:.4f}')
    return mean_dice, mean_iou


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "log/gta5_pretrain_2/718.pt"  # Path to the trained model

    from datasets.REFUGE import load_dataset

    # test_dataset = CustomDataset(image_folder, mask_folder, transform=transform)
    test_loader = load_dataset("_", "test")

    model = load_model(model_path, device).to(device)
    test_model(model, test_loader, device)
