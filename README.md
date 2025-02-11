# Real-Time Anomaly Segmentation for Road Scenes [[Course Project](https://docs.google.com/document/d/1ElljsAprT2qX8RpePSQ3E00y_3oXrtN_CKYC6wqxyFQ/edit?usp=sharing)]

This repository contains the code of the __Real-Time Anomaly Segmentation for Road Scenes__ project of the __Advanced Machine Learning__ course (A.Y. 2024/25) - Politecnico di Torino

## Repository structure
For instructions, please refer to the README in each folder:

- **[train](/train)**: contains tools for training the networks (ERFNet, ENet, BiSeNETv1) for the semantic segmentation.
- **[eval](/eval)**: contains tools for evaluating/visualizing the networkss outputs and performing anomaly segmentations.
    - **MSP (Maximum Softmax Probability)**
    - **MaxLogit**
    - **MaxEntropy**
    - **Temperature Scaling** 
- **[save](/save)**: contains
    -  trained BiSeNETv1, ERFNet and ENet using Cityscapes (19 known classes and 1 void class) dataset with calculated weights. 
    -  trained ERFNet (with pretrained encoder in ImageNet) using Cityscapes (19 known classes and 1 void class).
    -  trained ERFNet with pretrained encoder by using
        - Enhanced Isotropy Maximization loss, 
        - Logit Normalization loss,
        - Enhanced Isotropy Maximization loss + Focal loss,
        - Enhanced Maximization loss + Cross Entropy loss,
        - Logit Normalization loss + Focal loss,
        - Logit Normalization loss + Cross Entropy loss
- **[imagenet](/imagenet)**: contains script and model for pretraining ERFNet's encoder in Imagenet.
- **[trained_models](/trained_models)**: contains the trained models used in the papers. 
- **[project#6_colab.ipynb](/project#6_colab.ipynb)**: Jupyter Notebook compatible with Google Colab to set up Colan environment, donwload datasets, evaluate anomalies, train models.

## Datasets:

- **To train and validation**:
    - [**Cityscapes dataset**](https://www.cityscapes-dataset.com/): used **gtFine** for labels and **leftImg8bit** for RGB images. We have used [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to convert labels to `_labelTrainIds`.
- **To test**:
    - [**SegmentMeIfYouCan**](https://segmentmeifyoucan.com/datasets)
        - **RoadAnomaly21**: images are png, labels' masks are png
        - **RoadObstacle21**: images are webp, labels' masks are png
    - [**Fishscapes**](https://fishyscapes.com/dataset)
        - **FS Static**: images are jpg, labels' masks are png
        - **FS Lost and Found**: images are png, labels' masks are png
    - [**Road Anomaly**](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)
        - **RoadAnomaly**:images are jpg, labels' masks are png


## Requirements:

* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0). 
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)