# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GTADataset(CustomDataset):
    # Define classes and palette based on REFUGE dataset
    CLASSES = ('background', 'optic_disc', 'optic_cup')  # Example classes
    PALETTE = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # Corresponding colors

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train', 'val', 'test']  # Adjusted for REFUGE splits
        super(GTADataset, self).__init__(
            img_suffix='.jpg',  # REFUGE images might be '.jpg'
            seg_map_suffix='.png',  # Adjust based on REFUGE mask format
            split=None,
            **kwargs)


#
# @DATASETS.register_module()
# class GTADataset(CustomDataset):
#     # CLASSES = CityscapesDataset.CLASSES
#     # PALETTE = CityscapesDataset.PALETTE
#
#     CLASSES = ('background', 'optic_disc', 'optic_cup')  # Example classes
#     PALETTE = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # Corresponding colors
#
#     def __init__(self, **kwargs):
#         assert kwargs.get('split') in [None, 'train']
#         if 'split' in kwargs:
#             kwargs.pop('split')
#
#         super(GTADataset, self).__init__(
#             img_suffix='.png',
#             seg_map_suffix='_labelTrainIds.png',
#             split=None,
#             **kwargs)

