# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .rotatedText import RotatedTextDataset
from .transforms import *  # noqa: F401, F403
from .MLT import MLTDataset
from  .SAR import SARDataset

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset','RotatedTextDataset','MLTDataset','SARDataset'
]
