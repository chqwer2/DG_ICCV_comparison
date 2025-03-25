# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
from idlelib import filelist

# from Past_to_Future.our.control.ldm.models.diffusion.dpm_solver.dpm_solver import expand_dims

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from hgformer import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    def update_num_classes(cfg):
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if key == 'NUM_CLASSES':
                    cfg[key] = 3  # TODO
                else:
                    update_num_classes(value)
        elif isinstance(cfg, list):
            for item in cfg:
                update_num_classes(item)

        # Call the function on your config

    update_num_classes(cfg)
    
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


import numpy as np


def compute_overall_iou_dice(gt, pred, ignore_class = 0):
    """
    Compute overall IoU and Dice score for the entire image, ignoring the background class.

    Parameters:
        gt (ndarray): Ground truth mask of shape (H, W)
        pred (ndarray): Predicted mask of shape (H, W)
        ignore_class (int): Class ID to ignore (default: 0 for background)

    Returns:
        tuple: (Overall IoU, Overall Dice)
    """
    # Create a mask to ignore background pixels
    fg_mask = gt != ignore_class  # True for foreground pixels

    # Apply the mask to filter out background pixels
    gt = gt[fg_mask]
    pred = pred[fg_mask]

    # Compute True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = np.sum(gt == pred)  # Correctly classified pixels
    fp = np.sum(gt != pred)  # Incorrectly classified pixels
    fn = fp  # Every misclassification is both FP and FN

    # Compute overall IoU
    overall_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # Compute overall Dice coefficient
    overall_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return overall_iou, overall_dice


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # import ipdb; ipdb.set_trace()
    # filelist = GetFileFromThisRootDir(args.input[0])
    refuge_root = "/home/cbtil3/Downloads/REFUGE"
    split="val"
    img_dir  = os.path.join(refuge_root, f"{split}/Images")
    mask_dir = os.path.join(refuge_root, f"{split}/Masks")

    img_files  = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    filelist = img_files
    num_classes = 3

    all_iou = []
    all_dice = []

    import matplotlib.pyplot as plt
    for (path, mask_path) in tqdm.tqdm(zip(filelist, mask_files), disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        msk = plt.imread(mask_path) * 255

        img = cv2.resize(img, (512, 512))
        msk = cv2.resize(msk, (512, 512), interpolation=cv2.INTER_NEAREST)

        start_time = time.time()
        # predictions, visualized_output = demo.run_on_image(img)
        predictions = demo.predictor(img)

        # import ipdb; ipdb.set_trace()
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        basename = os.path.basename(path)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        output_path = os.path.join(args.output, basename)

        outimg = predictions['sem_seg'].detach().cpu().numpy().argmax(0).astype(np.uint8)
        outimg = cv2.resize(outimg, (512, 512), interpolation=cv2.INTER_NEAREST)


         # = calculate_iou(msk, outimg)
        iou, dice = compute_overall_iou_dice(msk, outimg)
        all_iou.append(iou)
        all_dice.append(dice)

        print("outimg shape:", outimg.shape, np.unique(outimg), iou, dice)
        # outimg shape: (1024, 1024)
        outimg = outimg*125
        outimg = cv2.cvtColor(outimg, cv2.COLOR_GRAY2BGR)

        # Calculate IoU and Dice
        print("output path=", output_path)
        cv2.imwrite(output_path, outimg)

print("IoU:", np.mean(all_iou, axis=0))
print("Dice:", np.mean(all_dice, axis=0))

