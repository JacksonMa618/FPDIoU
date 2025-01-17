[🛠️Installation](https://mmrotate.readthedocs.io/en/1.x/install.html) |

<!--中/英 文档切换-->

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMRotate is an open-source toolbox for rotated object detection based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.6+**.

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

<details open>
<summary><b>Major Features</b></summary>

- **Support multiple angle representations**

  MMRotate provides three mainstream angle representations to meet different paper settings.

- **Modular Design**

  We decompose the rotated object detection framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in rotated object detection.

</details>


| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

**v1.0.0rc1** was released in 30/12/2022:

- Support [RTMDet](configs/rotated_rtmdet) rotated object detection models. The technical report of RTMDet is on [arxiv](https://arxiv.org/abs/2212.07784)
- Support [H2RBox](configs/h2rbox) models. The technical report of H2RBox is on [arxiv](https://arxiv.org/abs/2210.06742)

## Installation

Please refer to [Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) for more detailed instruction.

## Getting Started

Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of MMRotate.

For detailed user guides and advanced guides, please refer to our [documentation](https://mmrotate.readthedocs.io/en/1.x/):

- User Guides
  - [Train & Test](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html#train-test)
    - [Learn about Configs](https://mmrotate.readthedocs.io/en/1.x/user_guides/config.html)
    - [Inference with existing models](https://mmrotate.readthedocs.io/en/1.x/user_guides/inference.html)
    - [Dataset Prepare](https://mmrotate.readthedocs.io/en/1.x/user_guides/dataset_prepare.html)
    - [Test existing models on standard datasets](https://mmrotate.readthedocs.io/en/1.x/user_guides/train_test.html)
    - [Train predefined models on standard datasets](https://mmrotate.readthedocs.io/en/1.x/user_guides/train_test.html)
    - [Test Results Submission](https://mmrotate.readthedocs.io/en/1.x/user_guides/test_results_submission.html)
  - [Useful Tools](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html#useful-tools)
- Advanced Guides
  - [Basic Concepts](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#basic-concepts)
  - [Component Customization](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#component-customization)
  - [How to](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#how-to)

We also provide colab tutorial [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/MMRotate_Tutorial.ipynb).

To migrate from MMRotate 0.x, please refer to [migration](https://mmrotate.readthedocs.io/en/1.x/migration.html).

The code of different losses are shown in mmrotate-1.x/mmrotate/models/losses


## Model Checkpoints
| Model                    | IoU Loss |   Checkpoint                                                 |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
|      | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## FAQ

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{MA2024105381,
title = {FPDIoU Loss: A loss function for efficient bounding box regression of rotated object detection},
journal = {Image and Vision Computing},
pages = {105381},
year = {2024},
issn = {0262-8856},
doi = {https://doi.org/10.1016/j.imavis.2024.105381},
url = {https://www.sciencedirect.com/science/article/pii/S0262885624004864},
author = {Siliang Ma and Yong Xu},
keywords = {Rotated object detection, Bounding box regression, Loss function, Minimum points distance},
abstract = {Bounding box regression is one of the important steps of object detection. However, rotation detectors often involve a more complicated loss based on SkewIoU which is unfriendly to gradient-based training. Most of the existing loss functions for rotated object detection calculate the difference between two bounding boxes only focus on the deviation of area or each points distance (e.g., LSmooth−L1, LRotatedIoU and LPIoU). The calculation process of some loss functions is extremely complex (e.g. LKFIoU). In order to improve the efficiency and accuracy of bounding box regression for rotated object detection, we proposed a novel metric for arbitrary shapes comparison based on minimum points distance, which takes most of the factors from existing loss functions for rotated object detection into account, i.e., the overlap or nonoverlapping area, the central points distance and the rotation angle. We also proposed a loss function called LFPDIoU based on four points distance for accurate bounding box regression focusing on faster and high quality anchor boxes. In the experiments, FPDIoU loss has been applied to state-of-the-art rotated object detection (e.g., RTMDET, H2RBox) models training with three popular benchmarks of rotated object detection including DOTA, DIOR, HRSC2016 and two benchmarks of arbitrary orientation scene text detection including ICDAR 2017 RRC-MLT and ICDAR 2019 RRC-MLT, which achieves better performance than existing loss functions.}
}
```
