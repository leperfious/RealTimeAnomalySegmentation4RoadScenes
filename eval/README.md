# Functions for evaluating/visualizing the network's output

Currently there are 7 usable functions to evaluate stuff:
- **[eval_iou.py](/eval/eval_iou.py)**
- **[evalAnomaly.py](/eval/evalAnomaly.py)**
- **[eval_cityscapes_color.py](/eval/eval_cityscapes_color.py)**
- **[eval_cityscapes_server.py](/eval/eval_cityscapes_server.py)**
- **[eval_forwardTime.py](/eval/eval_forwardTime.py)**



## **[eval_iou.py](/eval/eval_iou.py)**

This code can be used to calculate the IoU (mean and per-class) with baseliens MSP, MaxLogit, MaxEntropy in a subset of images with labels available, like Cityscapes val/train sets. Here we can choose temperature value to apply temperature scaling. It can be used for pretrained ERFNet, and trained BiSeNet, ENet, ERFNet on Cityscapes with 19 known classes and one void class.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val' or 'train'). For other options check the bottom side of the file.
**Networks:** Specify the network folder path with '--loadDir' option, weights with '--loadWeight', and model with 'loadModel'
**Temperature:** Specify temperature grade to apply by adding value to '--temperature '

**Examples:**
```
python eval_iou.py --loadDir ..\trained_models --loadWeights erfnet_pretrained.pth --loadModel erfnet.py --subset val --datadir ..\datasets\cityscapes --method msp
python eval_iou.py --loadDir ..\trained_models --loadWeights erfnet_pretrained.pth --loadModel erfnet.py --subset val --datadir ..\datasets\cityscapes --method max_logit
python eval_iou.py --loadDir ..\trained_models --loadWeights erfnet_pretrained.pth --loadModel erfnet.py --subset val --datadir ..\datasets\cityscapes --method max_entropy

python eval_iou.py --temperature 2 --method msp

```


## **[evalAnomaly.py](/eval/evalAnomaly.py)**

This code can be used to produce anomaly segmentation results on  anomaly metrics (FPR95, AuPRC) using Validation Datasets that mentioned.
**Temperature:** Specify temperature grade to apply by adding value to '--temperature '

**Examples:**
```
python evalAnomaly.py --method msp
python evalAnomaly.py --temperature 2 --method msp
python evalAnomaly.py --method max_logit
python evalAnomaly.py --method max_entropy

```


## **[eval_cityscapes_color.py](/eval/eval_cityscapes_color.py)**

This code can be used to produce segmentation of the Cityscapes images in color for visualization purposes. By default it saves images in eval/save_color/ folder. You can also visualize results in visdom with --visualize flag.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_color.py --datadir /content/datasets/cityscapes/ --subset val
```

## **[eval_cityscapes_server.py](/eval/eval_cityscapes_server.py)**

This code can be used to produce segmentation of the Cityscapes images and convert the output indices to the original 'labelIds' so it can be evaluated using the scripts from Cityscapes dataset (evalPixelLevelSemanticLabeling.py) or uploaded to Cityscapes test server. By default it saves images in eval/save_results/ folder.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_server.py --datadir /content/datasets/cityscapes/ --subset val
```



## **[eval_forwardTime.py](/eval/eval_forwardTime.py)**
This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution. 

**Options:** Option '--width' specifies the width (default: 1024). Option '--height' specifies the height (default: 512). For other options check the bottom side of the file.

**Examples:**
```
python eval_forwardTime.py
```

**NOTE**: Paper values were obtained with a single Titan X (Maxwell) and a Jetson TX1 using the original Torch code. The pytorch code is a bit faster, but cudahalf (FP16) seems to give problems at the moment for some pytorch versions so this code only runs at FP32 (a bit slower).


**NOTE**: In the network modules, Net has been changed to the ENet, ERFNet, BiSeNet
